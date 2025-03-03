import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.modules import *
from losses.homography import *

# code from RC-MVSNet
class UnSupLoss(nn.Module):
    def __init__(self):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth, stage_idx):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]

        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # depth reshape
        # depth = depth.unsqueeze(dim=1)  # [B, 1, H, W]
        # depth = F.interpolate(depth, size=[img_height, img_width])
        # depth = depth.squeeze(dim=1)  # [B, H, W]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25,recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5,recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.18 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss

class UnSupLoss_no_smooth(nn.Module):
    def __init__(self):
        super(UnSupLoss_no_smooth, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth, stage_idx):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]
        # ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        # 按照stage进行resize，匹配到每个阶段的分辨率
        # 这里尽量不要使用bilinear，这个会平滑图像的边缘，可能会对自监督损失有影响
        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # depth reshape
        # depth = depth.unsqueeze(dim=1)  # [B, 1, H, W]
        # depth = F.interpolate(depth, size=[img_height, img_width])
        # depth = depth.squeeze(dim=1)  # [B, H, W]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        # self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25,recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5,recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        # self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss

class UnSupLoss_07(nn.Module):
    def __init__(self):
        super(UnSupLoss_07, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth, stage_idx):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]
        # ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        # 按照stage进行resize，匹配到每个阶段的分辨率
        # 这里尽量不要使用bilinear，这个会平滑图像的边缘，可能会对自监督损失有影响
        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # depth reshape
        # depth = depth.unsqueeze(dim=1)  # [B, 1, H, W]
        # depth = F.interpolate(depth, size=[img_height, img_width])
        # depth = depth.squeeze(dim=1)  # [B, H, W]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25,recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5,recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.19 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss

class UnSupLoss_06(nn.Module):
    def __init__(self):
        super(UnSupLoss_06, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth, stage_idx):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]
        # ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        # 按照stage进行resize，匹配到每个阶段的分辨率
        # 这里尽量不要使用bilinear，这个会平滑图像的边缘，可能会对自监督损失有影响
        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # depth reshape
        # depth = depth.unsqueeze(dim=1)  # [B, 1, H, W]
        # depth = F.interpolate(depth, size=[img_height, img_width])
        # depth = depth.squeeze(dim=1)  # [B, H, W]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25,recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5,recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.16 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss

class UnsupLossMultiStage_06(nn.Module):
    def __init__(self):
        super(UnsupLossMultiStage_06, self).__init__()
        self.unsup_loss = UnSupLoss_06()

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)


            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs

class UnsupLossMultiStage_07(nn.Module):
    def __init__(self):
        super(UnsupLossMultiStage_07, self).__init__()
        self.unsup_loss = UnSupLoss_07()

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)


            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs

class UnsupLossMultiStage(nn.Module):
    def __init__(self):
        super(UnsupLossMultiStage, self).__init__()
        self.unsup_loss = UnSupLoss()

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)


            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs

class UnsupLossMultiStage_no_smooth(nn.Module):
    def __init__(self):
        super(UnsupLossMultiStage_no_smooth, self).__init__()
        self.unsup_loss = UnSupLoss_no_smooth()

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)


            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            # scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs


# code from KD-MVS
class unsup_loss(nn.Module):
    def __init__(self):
        super(unsup_loss, self).__init__()
        self.ssim = SSIM()

    def forward(self, inputs, imgs, sample_cams, num_views=5, **kwargs):
        # def unsup_loss(inputs, imgs, sample_cams, num_views=5, **kwargs):

        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)
        total_photo_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)
        total_feature_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)

        reconstr_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)
        ssim_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)
        smooth_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)

        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            depth_est = stage_inputs["depth"].unsqueeze(1)   # b,1,h,w
            features = stage_inputs['features']

            log_var = stage_inputs['var']   # b,h,w

            ref_img = imgs[:,0] # b,c,h,w
            scale = depth_est.shape[-1] / ref_img.shape[-1]
            ref_img = F.interpolate(ref_img, scale_factor=scale, mode='bilinear', align_corners=True)
            ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            ref_cam = sample_cams[stage_key][:,0]   # b,2,4,4

            ref_feature = features[0].detach()  # b,c,h,w
            ref_feature = ref_feature.permute(0, 2, 3, 1)

            warped_img_list = []
            warped_feature_list = []
            feature_mask_list = []
            mask_list = []
            reprojection_losses = []
            fea_reprojection_losses = []

            for view in range(1, num_views):
                view_img = imgs[:,view]
                view_feature = features[view].detach()
                view_feature = view_feature.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
                view_cam = sample_cams[stage_key][:,view]
                view_img = F.interpolate(view_img, scale_factor=scale, mode='bilinear', align_corners=True)
                view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
                # warp view_img to the ref_img using the dmap of the ref_img
                warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth_est)
                warped_img_list.append(warped_img)
                mask_list.append(mask)

                warped_fea, fea_mask = inverse_warping(view_feature, ref_cam, view_cam, depth_est)
                warped_feature_list.append(warped_fea)
                feature_mask_list.append(fea_mask)

                reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
                fea_reconstr_loss = compute_reconstr_loss(warped_fea, ref_feature, fea_mask, simple=False)
                valid_mask = 1 - mask  # replace all 0 values with INF
                fea_valid_mask = 1 - fea_mask
                reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)  # 这有什么作用
                fea_reprojection_losses.append(fea_reconstr_loss + 1e4 * fea_valid_mask)

                # SSIM loss##
                if view < 3:
                    ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))
            del features, view_feature, ref_feature

            ##smooth loss##
            smooth_loss += depth_smoothness(depth_est.unsqueeze(dim=-1), ref_img, 1.0)
            # top-k operates along the last dimension, so swap the axes accordingly
            reprojection_volume = torch.stack(reprojection_losses).permute(1,2,3,4,0)  # [4, 128, 160, 1, 6]
            top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
            top_vals = torch.neg(top_vals)
            top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
            top_mask = top_mask.float()
            top_vals = torch.mul(top_vals, top_mask)  # [4, 128, 160, 1, 3]
            top_vals = torch.sum(top_vals, dim=-1)  # [4, 128, 160, 1]
            top_vals = top_vals.permute(0, 3, 1, 2)  # [4, 1, 128, 160]

            fea_reprojection_volume = torch.stack(fea_reprojection_losses).permute(1,2,3,4,0)  # [4, 128, 160, 1, 6]
            fea_top_vals, fea_top_inds = torch.topk(torch.neg(fea_reprojection_volume), k=3, sorted=False)
            fea_top_vals = torch.neg(fea_top_vals)
            fea_top_mask = fea_top_vals < (1e4 * torch.ones_like(fea_top_vals, device=device))
            fea_top_mask = fea_top_mask.float()
            fea_top_vals = torch.mul(fea_top_vals, fea_top_mask)  # [4, 128, 160, 1, 3]
            fea_top_vals = torch.sum(fea_top_vals, dim=-1)  # [4, 128, 160, 1]
            fea_top_vals = fea_top_vals.permute(0, 3, 1, 2)  # [4, 1, 128, 160]

            loss1 = torch.mean(torch.exp(-log_var) * top_vals)
            loss2 = torch.mean(log_var)
            loss3 = torch.mean(torch.exp(-log_var) * fea_top_vals)
            reconstr_loss = 0.5 * (loss1 + 0.25*loss3 + 0.1 * loss2)

            # self.reconstr_loss = torch.mean()
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += (12 * 2 * reconstr_loss + 6 * ssim_loss + 0.18 * smooth_loss) * depth_loss_weights[stage_idx]
            total_photo_loss += 12*loss1* depth_loss_weights[stage_idx]
            total_feature_loss += 3*loss3* depth_loss_weights[stage_idx]
            # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15

        return total_loss, 12*2*reconstr_loss, 6*ssim_loss, \
                0.18*smooth_loss, total_photo_loss, total_feature_loss


# code from JD-CAS
class UnSupLoss(nn.Module):
    def __init__(self):
        super(UnSupLoss, self).__init__()
        # self.ssim = SSIM()

    def forward(self, imgs, cams, depth):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]
        # ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        # ref_img = F.interpolate(ref_img, size=[depth.shape[1], depth.shape[2]])
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            # view_img = F.interpolate(view_img, size=[depth.shape[1], depth.shape[2]])
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1,2,3,4,0)
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals = torch.neg(top_vals)
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        # self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.18 * self.smooth_loss
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.05 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15

        return self.unsup_loss

