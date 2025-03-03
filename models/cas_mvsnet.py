import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .FMT import FMT_with_pathway
from models.geometry import GeoFeatureFusion, GeoRegNet2d

Align_Corners_Range = False

class DepthNet(nn.Module):
    def __init__(self, mode="adaptive", in_channels=None):
        super(DepthNet, self).__init__()
        self.mode = mode
        assert mode in ("variance", "adaptive"), "Don't support {}!".format(mode)
        if self.mode == "adaptive":
            self.weight_net = nn.ModuleList([AggWeightNetVolume(in_channels[i]) for i in range(len(in_channels))])

    def forward(self, stage_idx, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        # volume_sum = ref_volume
        # volume_sq_sum = ref_volume ** 2
        # del ref_volume
        if self.mode == "variance":
            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2
            del ref_volume
        elif self.mode == "adaptive":
            volume_adapt = None
        
        
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            # warped_volume = homo_warping(src_fea, src_proj[:, 2], ref_proj[:, 2], depth_values)

            # variance for costvolume
            # if self.training:
            #     volume_sum = volume_sum + warped_volume
            #     volume_sq_sum = volume_sq_sum + warped_volume ** 2
            # else:
            #     # TODO: this is only a temporal solution to save memory, better way?
            #     volume_sum += warped_volume
            #     volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            # del warped_volume

            if self.mode == "variance":
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            elif self.mode == "adaptive":
                # (b,c,d,h,w)
                warped_volume = (ref_volume - warped_volume).pow_(2)
                # print("&&&&&&&&&&&********")
                # (B,C,D,H,W)
                #  torch.Size([8, 32, 48, 128, 160])、 torch.Size([8, 16, 32, 256, 320])、torch.Size([8, 8, 8, 512, 640])
                # print("the warped_volume size is: {}".format(warped_volume.shape))
                weight = self.weight_net[stage_idx](warped_volume)  # (b, 1, d, h, w)
                # print("*** &&&  the shape of weight is: ", weight.shape)
                if volume_adapt is None:
                    volume_adapt = (weight + 1) * warped_volume
                else:
                    volume_adapt = volume_adapt + (weight + 1) * warped_volume
            del warped_volume


        # aggregate multiple feature volumes by variance
        # volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            
        # aggregate multiple feature volumes by variance or adaptive
        if self.mode == "variance":
            volume_agg = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        elif self.mode == "adaptive":
            volume_agg = volume_adapt / (num_views - 1)
        # print("********")
        # （B,C,D,H,W）
        # torch.Size([8, 32, 48, 128, 160])、 torch.Size([8, 16, 32, 256, 320])、 torch.Size([8, 8, 8, 512, 640])
        # print("the volume_agg size is: {}".format(volume_agg.shape))

        # step 3. cost volume regularization
        # cost_reg = cost_regularization(volume_variance)
        cost_reg = cost_regularization(volume_agg)   # （B,C,Ndepth,H,W）-> (B,1,Ndepth,H,W)
        #torch.Size([4, 1, 48, 128, 160]) torch.Size([4, 1, 32, 128, 160]) torch.Size([4, 1, 8, 128, 160])
        # print("**** the shape of cost_reg is: ", cost_reg.shape) 

        # # aggregate multiple feature volumes by adaptive
        # volume_adaptive = volume_adapt / (num_views - 1)
        # # cost volume regularization
        # cost_reg = cost_regularization(volume_adaptive)

        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        prob_volume_pre = cost_reg.squeeze(1)  # (B,Ndepth,H,W)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)  ## 将深度信息压缩为0~1之间的分布，得到概率体
        depth = depth_regression(prob_volume, depth_values=depth_values) 

        with torch.no_grad():
            # photometric confidence  (B,H,W)
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        # add from me
        samp_variance = (depth_values - depth.unsqueeze(1)) ** 2
        # default setting: 1.5*sigma   3*sigma->99% 2*sigma->95% sigma->68%
        # exp_variance = 1.5 * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5
        exp_variance = 3 * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # # torch.Size([8, 128, 160]) torch.Size([8, 256, 320]) torch.Size([8, 512, 640])
        # print("the photometric_confidence size is: ", photometric_confidence.shape )  

        # from me
        # return {"depth": depth,  "photometric_confidence": photometric_confidence, 'variance': exp_variance}
        # return {"depth": depth,  "photometric_confidence": photometric_confidence}
        # from TransMVSNet 20240328
        return {"depth": depth,  "photometric_confidence": photometric_confidence, 
                'variance': exp_variance, "prob_volume": prob_volume, "depth_values": depth_values}


class CascadeMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=[64, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8], agg_mode="adaptive"):  # 修改了各层假设深度数量，由（48,32,8）——>(64,32,8)
        super(CascadeMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)

        
        self.GeoFeatureFusionNet = GeoFeatureFusion(
            convolutional_layer_encoding="z", mask_type="basic", add_origin_feat_flag=True)

        # from TransMVSNet
        # self.FMT_with_pathway = FMT_with_pathway()

        self.geo_reg_encodings = ['std', 'z', 'z', 'z']     # must use std in idx-0

        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()
        # self.DepthNet = DepthNet()
        # 其中加入cost volume聚合方式：自适应权重生成网络   3 stage,feature.out_channels: [32,16,8]
        # print("$$$$$$$$$$  the feature.out_channels is: {}".format(self.feature.out_channels))
        self.DepthNet = DepthNet(agg_mode, self.feature.out_channels)

    def forward(self, imgs, proj_matrices, depth_values,intrinsics_matrices):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)
        # print("***********&&&&&&&&&&")
        # print("the size of img is: {}".format(imgs.shape))   # torch.Size([7, 5, 3, 512, 640])   torch.Size([2, 7, 3, 576, 768])
        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]  # (B,C,H,W)
            features.append(self.feature(img))

        # from TransMVSNet
        # features = self.FMT_with_pathway(features) 
        
        # print("************")
        outputs = {}
        # depth, cur_depth = None, None
        depth, cur_depth, exp_var = None, None, None
        for stage_idx in range(self.num_stage):
            #stage feature, proj_mats, scales
            stage_name = "stage{}".format(stage_idx + 1)
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            intrinsics_matrices_stage = intrinsics_matrices[stage_name]

            
            # @Note features fusion
            if stage_idx >= 1:
                ref_img_stage = F.interpolate(imgs[:,0], size=None, scale_factor=1./2**(2-stage_idx), mode="bilinear", align_corners=False)
                depth_last = F.interpolate(depth_last.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False)
                confidence_last = F.interpolate(confidence_last.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False)

                # print("*****************&&&&&&&&&&&&&&&&&&&")
                # print("the shape of ref_img_stage is {}".format(ref_img_stage.shape))   # torch.Size([8, 3, 256, 320])  torch.Size([8, 3, 512, 640])
                # print("the shape of depth_stage is {}".format(depth_last.shape))   # torch.Size([8, 1, 256, 320])  torch.Size([8, 1, 512, 640])
                # print("the shape of confidence_stage is {}".format(confidence_last.shape))  # torch.Size([8, 1, 256, 320])  torch.Size([8, 1, 512, 640])

                # reference feature
                features_stage[0] = self.GeoFeatureFusionNet(
                    ref_img_stage, depth_last, confidence_last, depth_values,
                    stage_idx, features_stage[0], intrinsics_matrices_stage
                )

                # print("the shape of features_stage[0] is {}".format(features_stage[0].shape))


            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                    exp_var = exp_var.detach()   # add from me
                else:
                    cur_depth = depth
                # CasMVSNet原代码中cur_depth处理有squeeze过程，所以最后输出的大小为(B,H,W)
                # 但是UCSMVSNet这一步没有squeeze处理过程，所以最后输出的大小为（B,1,H,W）
                # UCSMVSNet为什么没有这个squeeze处理过程，主要是后面的uncertainty_aware_samples一些代码设置原因
                # 但无论怎样，最后深度采样模块处理完之后的depth_samples大小都是(B,D,H,W) 
                # cur_depth = F.interpolate(cur_depth.unsqueeze(1), [img.shape[2], img.shape[3]], mode='bilinear',
                #                                 align_corners=Align_Corners_Range).squeeze(1)
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                [img.shape[2], img.shape[3]], mode='bilinear',
                                                align_corners=Align_Corners_Range)
                exp_var = F.interpolate(exp_var.unsqueeze(1),[img.shape[2], img.shape[3]], mode='bilinear')  # add from me
                
            else:
                cur_depth = depth_values
            

            # depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
            #                                             ndepth=self.ndepths[stage_idx],
            #                                             depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
            #                                             dtype=img[0].dtype,
            #                                             device=img[0].device,
            #                                             shape=[img.shape[0], img.shape[2], img.shape[3]],
            #                                             max_depth=depth_max,
            #                                             min_depth=depth_min)
            
            # (B,D,H,W)
            depth_range_samples = uncertainty_aware_samples(cur_depth=cur_depth,
                                                        exp_var=exp_var,
                                                        ndepth=self.ndepths[stage_idx],
                                                        dtype=img[0].dtype,
                                                        device=img[0].device,
                                                        shape=[img.shape[0], img.shape[2], img.shape[3]])
            if stage_idx == 2:
                print("***** the depth range samples is: ")
                #  DTU： torch.Size([4, 8, 864, 1152])   TT：torch.Size([2, 8, 1056, 1920])  
                print("the shape of depth_range_samples is: ", depth_range_samples.shape) # (B,D,H,W)   
                print("**** shape ", depth_range_samples[0,:,575,1018].shape)  # **** shape  torch.Size([8])
                # print("**** ", depth_range_samples[0,:,575,1018])  
                # print("%%%% ", depth_range_samples[0,:,541,919])  
                # print("￥￥￥￥ ", depth_range_samples[0,:,577,961])  
                print("######### ", depth_range_samples[0,:,33,369])  
                print("@@@@@@", depth_range_samples[0,:,151,441])  
                print("!!!! ", depth_range_samples[0,:,106,390])  
                # ****  tensor([4.3179, 5.0203, 5.7246, 6.4333, 7.1521, 7.8951, 8.6949, 9.6286],device='cuda:0')
            # print("&&&&&&&&&&&&&&&&&&&&&&&&") 
            # torch.Size([8, 48, 512, 640])   torch.Size([8, 32, 512, 640])  torch.Size([8, 8, 512, 640])                                      
            # print("the size of depth_range_samples is: ", depth_range_samples.shape)

            # 1229:对cost volume aggregation加上stage_idx
            outputs_stage = self.DepthNet(stage_idx, features_stage, proj_matrices_stage,
                                          depth_values=F.interpolate(depth_range_samples.unsqueeze(1),
                                                                     [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], 
                                                                     mode='trilinear',
                                                                     align_corners=Align_Corners_Range).squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx])

            depth = outputs_stage['depth']
            depth_last = outputs_stage['depth']
            confidence_last = outputs_stage['photometric_confidence']
            exp_var = outputs_stage['variance']  # add from me


            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
            # dict_keys(['stage1', 'depth', 'photometric_confidence', 'variance', 'prob_volume', 'depth_values'])
            # dict_keys(['stage1', 'depth', 'photometric_confidence', 'variance', 'prob_volume', 'depth_values', 'stage2'])
            # dict_keys(['stage1', 'depth', 'photometric_confidence', 'variance', 'prob_volume', 'depth_values', 'stage2', 'stage3'])
            # print("******* the key of outputs is: {}".format(outputs.keys()))
            

        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs