import os
import cv2
import re
import signal
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool
from plyfile import PlyData, PlyElement

# 自己将read_pfm从dataset.data_io中单独复制过来了
# from datasets.data_io import read_pfm
from filter.tank_test_config import tank_cfg

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


# save a binary mask
def save_mask(filename, mask):
    # assert mask.dtype == np.bool   # AttributeError: module 'numpy' has no attribute 'bool'.
    assert mask.dtype == np.bool_
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected[2:3][K_xyz_reprojected[2:3]==0] += 0.00001
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(args, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = None
    masks = []
    for i in range(2, 11):
        # mask = np.logical_and(dist < i / 4, relative_depth_diff < i / 1300)
        mask = np.logical_and(dist < i * args.dist_base, relative_depth_diff < i * args.rel_diff_base)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src

# from geomvsnet
def scale_input(intrinsics, img):
    # if args.img_mode == "crop":
    #     intrinsics[1,2] = intrinsics[1,2] - 28  # 1080 -> 1024
    #     img = img[28:1080-28, :, :]
    # elif args.img_mode == "resize": 
    #     height, width = img.shape[:2]
    #     img = cv2.resize(img, (width, 1024))
    #     scale_h = 1.0 * 1024 / height
    #     intrinsics[1, :] *= scale_h
    height, width = img.shape[:2]
    img = cv2.resize(img, (width, 1024))
    scale_h = 1.0 * 1024 / height
    intrinsics[1, :] *= scale_h

    return intrinsics, img


def filter_depth(args, pair_folder, scan_folder, out_folder, plyfilename):
    num_stage = len(args.ndepths)
    print("*** begin begin begin filter_depth in neibu ***", flush=True)
    pair_file = os.path.join(pair_folder, "pair.txt")  # the pair file
    # for the final point cloud
    vertexs = []
    vertex_colors = []
    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    print("&&& it is using ", flush=True)
    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # src_views = src_views[:args.num_view]
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image, 都是从输出的文件夹中读取img/depth_est/confidence/cam
        # 而GeoMVSNet则是直接从原始输入中读取，所以可以看到其代码中又有一步后处理
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # print("*****TT: the size of ref_img is: ", ref_img.shape)  
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # print("*****TT: the size of ref_depth_est is: ", ref_depth_est.shape)  # (1024, 2048)  (1024, 1920)
        # load the photometric mask of the reference view
        # unimvsnet这种做法是读取了三阶段图像的confidence然后得到photo_mask,但一般网络最后只输出stage3的处理结果
        # np.logical_and为numpy库中的逻辑与运算
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        confidence2 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage2.pfm'.format(ref_view)))[0]
        confidence1 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage1.pfm'.format(ref_view)))[0]
        photo_mask = np.logical_and(np.logical_and(confidence > args.conf[2], confidence2 > args.conf[1]), confidence1 > args.conf[0])
        # print("*****TT: the size of confidence is: ", confidence.shape)   # (1024, 2048) (1024, 1920)

        # from me:只用最终阶段的photo_mask
        # photo_mask = confidence > args.conf[2]
        # ***** the prob_confidence is:  0.18  
        # print("***** the prob_confidence is: ", args.conf[2]) 
        # print("*****TT: the size of photo_mask is: ", photo_mask.shape)   # (1024, 2048) (1024, 1920)


        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        dy_range = len(src_views) + 1
        geo_mask_sums = [0] * (dy_range - 2)
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            

            masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(args, ref_depth_est, ref_intrinsics,
                                                                                               ref_extrinsics, src_depth_est,
                                                                                               src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            for i in range(2, dy_range):
                geo_mask_sums[i - 2] += masks[i - 2].astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least args.thres_view source views matched
        geo_mask = geo_mask_sum >= dy_range
        for i in range(2, dy_range):
            geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - 2] >= i)

        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        # 注释掉
        # print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
        #                                                                             photo_mask.mean(),
        #                                                                             geo_mask.mean(), final_mask.mean()))

        # if args.display:
        #     import cv2
        #     cv2.imshow('ref_img', ref_img[:, :, ::-1])
        #     cv2.imshow('ref_depth', ref_depth_est / 800)
        #     cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
        #     cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
        #     cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
        #     cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        # print("valid_points", valid_points.mean())  # 注释掉
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset

        # if num_stage == 1:
        #     color = ref_img[1::4, 1::4, :][valid_points]
        # elif num_stage == 2:
        #     color = ref_img[1::2, 1::2, :][valid_points]
        # elif num_stage == 3:
        #     color = ref_img[valid_points]

        color = ref_img[valid_points]
        # color_1 = ref_img[valid_points]
        # color_1 = ref_img[:,:,:][valid_points]
        # from geomvsnet, 后添加
        # color_1 = ref_img[28:1080-28, :, :][valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    print("&&& it is using using" ,flush=True)
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    print("&&& it is using using using" ,flush=True)
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    print("**** begin save the ply_model of: {}".format(plyfilename), flush=True)
    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename, flush=True)


def dypcd_filter_worker(args, scene):
    # ValueError: invalid literal for int() with base 10: 'thouse'
    # 需要传入int类型的数据传入了其他类型的数据
    # if args.testlist != "all":
    #     scan_id = int(scene[4:])
    #     save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)  
    # else:
    #     save_name = '{}.ply'.format(scene)
    print("&&& it is using dypcd_filter_worker")
    save_name = '{}.ply'.format(scene)  # GeoMVSNet也是直接这种做法
    pair_folder = os.path.join(args.datapath, scene)
    scan_folder = os.path.join(args.outdir, scene)  # 从这可以看出，其都是从输出文件夹中读取数据
    out_folder = os.path.join(args.outdir, scene)

    if scene in tank_cfg.scenes:
        scene_cfg = getattr(tank_cfg, scene)
        args.conf = scene_cfg.conf
    print("$$$$ begin filter_depth")
    print("the name of save_name is: {}".format(save_name))
    filter_depth(args, pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# 通过进程池来进行多线程处理数据，但似乎运行有问题         testlist中都是各场景的名字，[scan1, scan2, ....]
# 为什么调用不起来呢？
# def dypcd_filter(args, testlist, number_worker):
#     partial_func = partial(dypcd_filter_worker, args)
#     print("***begin: it is using dypcd_filter *****")  # 执行这一步
#     p = Pool(number_worker, init_worker)    # 卡在这一步
#     print("***after: it is using dypcd_filter *****")   # 这一步没有执行，说明卡在上面那一步
#     try:
#         print("&&& it is using p.map")   
#         p.map(partial_func, testlist)
#     except KeyboardInterrupt:
#         print("....\nCaught KeyboardInterrupt, terminating workers")
#         p.terminate()
#     else:
#         p.close()
#     print("*** it is wait pool")
#     p.join()


# def dypcd_filter(args, testlist):
#     # dypcd_filter_worker(args, scene)
#     partial_func = partial(dypcd_filter_worker, args)  # 提前输入一些固定参数
#     print("***begin: it is using dypcd_filter *****")
#     for scan in testlist:
#         partial_func(scan)


def dypcd_filter(args, testlist):
    print("*** it is using dypcd_filter", flush=True)
    for scene in testlist:
        save_name = '{}.ply'.format(scene)  # GeoMVSNet也是直接这种做法
        pair_folder = os.path.join(args.datapath, scene)
        scan_folder = os.path.join(args.outdir, scene)  # 从这可以看出，其都是从输出文件夹中读取数据
        out_folder = os.path.join(args.outdir, scene)

        if scene in tank_cfg.scenes:
            scene_cfg = getattr(tank_cfg, scene)
            args.conf = scene_cfg.conf
        print("$$$$ begin filter_depth", flush=True)
        print("the name of save_name is: {}".format(save_name))
        filter_depth(args, pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))
    