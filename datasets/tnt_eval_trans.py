from re import T
from torch.utils.data import Dataset
import numpy as np
import os, cv2
from PIL import Image
from datasets.data_io import *

# Test any dataset with scale and center crop
s_h, s_w = 0, 0
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.0,
                max_h=704,max_w=1280, inverse_depth=False, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.fix_res = kwargs.get("fix_res", True)  #whether to fix the resolution of input image.
        self.fix_wh = False
        self.max_h=max_h
        self.max_w=max_w
        self.inverse_depth=inverse_depth
        self.scans = listfile

        self.image_sizes = {'Family': (1920, 1080),
                            'Francis': (1920, 1080),
                            'Horse': (1920, 1080),
                            'Lighthouse': (2048, 1080),
                            'M60': (2048, 1080),
                            'Panther': (2048, 1080),
                            'Playground': (1920, 1080),
                            'Train': (1920, 1080),
                            'Auditorium': (1920, 1080),
                            'Ballroom': (1920, 1080),
                            'Courtroom': (1920, 1080),
                            'Museum': (1920, 1080),
                            'Palace': (1920, 1080),
                            'Temple': (1920, 1080)}

        assert self.mode == "test"
        self.metas = self.build_list()
        print('Data Loader : data_eval_T&T**************' )

    # def build_list(self):
    #     metas = []
    #     scans = self.scans
    #     for scan in scans:
    #         pair_file = "{}/pair.txt".format(scan)
    #         # read the pair file
    #         with open(os.path.join(self.datapath, pair_file)) as f:
    #             num_viewpoint = int(f.readline())
    #             for view_idx in range(num_viewpoint):
    #                 ref_view = int(f.readline().rstrip())
    #                 src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
    #                 if len(src_views) == 0:
    #                     continue
    #                 metas.append((scan, ref_view, src_views))
    #     print("dataset", self.mode, "metas:", len(metas))
    #     return metas

    # 返回的metas有对每个场景设置默认的interval
    def build_list(self):
        metas = []
        scans = self.scans

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        # print("dataset", self.mode, "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    # from general_eval.py
    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        # cam.txt输入为4个参数的：depth_min,depth_interval,depth_nums,depth_max  (其中depth_min+depth_interval*depth_nums = depth_max)
        # cam.txt输入为2个参数的：depth_min,depth_interval
        if len(lines[11].split()) >= 3:  # 对TT和blendedmvs数据集最后一行的额外处理，因为其输入4个参数，而dtu输入2个参数
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)

        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def center_img(self, img): # this is very important for batch normalization
        img = img.astype(np.float32)
        var = np.var(img, axis=(0,1), keepdims=True)
        mean = np.mean(img, axis=(0,1), keepdims=True)
        return (img - mean) / (np.sqrt(var) )

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else: # 对于TT而言，直接运行到这步。 由（1920/2048，1080）——>(1920/2048,1056)
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base
        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))
        return img, intrinsics

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        # scan, ref_view, src_views = meta
        scan, ref_view, src_views, scene_name = meta
        # img_w, img_h = self.image_sizes[scan]

        if self.nviews>len(src_views):
              self.nviews=len(src_views)+1

        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = (self.read_img(img_filename))
            # imgs.append(self.read_img(img_filename))
            # intrinsics, extrinsics, depth_min, depth_interval, depth_max = self.read_cam_file(proj_mat_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=
                                                                                   self.interval_scale[scene_name])

            img, intrinsics = self.scale_mvs_input(img, intrinsics,  self.image_sizes[scan][0],  self.image_sizes[scan][1])
            # img, intrinsics = self.scale_mvs_input(img, intrinsics,  self.max_w,  self.max_h)   
            # print("******* after scale input, the size of img is : ", img.shape)  # (1056,1920/2048,3)


            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)


        imgs = np.stack(imgs).transpose([0, 3, 1, 2]) # B,C,H,W
        proj_matrices = np.stack(proj_matrices)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        intrinsics = np.stack(intrinsics)  # intrinsics of cam is same
        stage2_ins = intrinsics.copy()
        stage2_ins[:2, :] = intrinsics[:2, :] * 2.0
        stage3_ins = intrinsics.copy()
        stage3_ins[:2, :] = intrinsics[:2, :] * 4.0

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        intrinsics_matrices = {
            "stage1": intrinsics,
            "stage2": stage2_ins,
            "stage3": stage3_ins
        }


        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "intrinsics_matrices": intrinsics_matrices,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}