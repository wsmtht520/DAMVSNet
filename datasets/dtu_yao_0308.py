from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale

        # if len(lines[11].split()) >= 3:
        #     num_depth = lines[11].split()[2]
        #     depth_max = depth_min + int(float(num_depth)) * depth_interval
        #     depth_interval = (depth_max - depth_min) / self.total_depths

        # depth_interval *= self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

        # return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        # 将PIL库读取到的图像（属于PIL图像）转换为numpy格式
        np_img = np.array(img, dtype=np.float32) / 255.  
        return np_img
    
    # edge feature by sobel
    # def edge_extra(self, ori_image):
    #     numpy_image = ori_image * 255.  # 恢复回0~255，因为之前读取的时候归一化了
    #     # Convert the image to grayscale if it's a color image
    #     if len(numpy_image.shape) == 3:
    #         gray_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
    #     else:
    #         gray_image = numpy_image

    #     # Apply Sobel operator for edge detection
    #     sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    #     sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

    #     # Compute the magnitude of the gradient
    #     grad_img = np.sqrt(sobel_x**2 + sobel_y**2)
    #     grad_imgScale = grad_img / 255.   # scale 0~255 to 0~1
    #     return grad_imgScale

    def prepare_img(self, hr_img):
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

        #downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        #crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        # #downsample
        # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_depth_hr(self, filename):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        # print("***** ++++ *********")
        # print("***** the scan of meta is: ", scan)
        # print("***** the light_idx of meta is: ", light_idx)
        # print("***** the ref_view of meta is: ", ref_view)
        # print("***** the src_views of meta is: ", src_views)
        # print("***** the vied_ids is: ", view_ids)
        # print("@@@@@@@@@@@@@@@@@@@@@@")

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)


            img = self.read_img(img_filename)

            # 用sobel提取边缘特征
            # origin_img = img.copy()
            # sobel_img = self.edge_extra(origin_img)


            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)


            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr)

                #get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

            imgs.append(img)
            # edge_imgs.append(sobel_img)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])  # (N,H,W,C)  -> (N,C,H,W)
        # imgs_test = np.stack(imgs_test).transpose([0, 3, 1, 2])
        # print(imgs.shape)    # (5, 3, 512, 640)
        # edge_feat = np.stack(edge_imgs).transpose([0,1,2])  # (N,H,W)
        # edge_feature = edge_feat[:,:,:,None].transpose([0,3,1,2])  # (N,H,W,1) -> (N,1,H,W)
        # edge_imgs = np.stack(edge_imgs)[:,:,:,None].transpose([0,3,1,2])
        # print("***** &&&&&&& ")
        # print(edge_imgs.shape)  # (5, 1, 512, 640)
        # print(type(edge_imgs))  # <class 'numpy.ndarray'>

        #ms proj_mats
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
            "stage3": stage3_pjmats}
        
        intrinsics_matrices = {
            "stage1": intrinsics,
            "stage2": stage2_ins,
            "stage3": stage3_ins}


        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "intrinsics_matrices": intrinsics_matrices,
                "mask": mask }