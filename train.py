import argparse, os, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def  
# from models import *
from models.cas_mvsnet import *  # 方便导入
from utils import *
import torch.distributed as dist
import matplotlib.pyplot as plt
import cv2

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

# dataset setting, from unimvsnet
# parser.add_argument("--datapath", type=str)
# parser.add_argument("--dataset_name", type=str, default="dtu_yao", choices=["dtu_yao", "general_eval", "blendedmvs"])


parser.add_argument('--dataset', default='dtu_yao', help='select dataset', choices=["dtu_yao", "general_eval", "blendedmvs"])
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

# parser.add_argument('--batch_size', type=int, default=1, help='train batch size') # 原设置
parser.add_argument('--batch_size', type=int, default=2, help='train batch size')
# 搞清楚这个numdepth和下面的各层假设深度数ndepths什么区别
# 这个numdpeth也只是开始根据cam.txt中的参数用来设置第一阶段的假设深度范围，具体调整还是要看后面的各层假设深度设置
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values') 
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/test0401_depthN5_adapCost_cpc_64', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')


parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory') 
# parser.add_argument('--pin_m', default='true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
# parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')  # CasMVSNet所用的假设深度数
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--nviews', type=int, default=5, help='num of view')
# cr_base_chs是ｃost volume regularization过程中三阶段输入的通道数,这里设置为（8,8,8）
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex') # 默认为False
# parser.add_argument('--using_apex', action='store_true', default=True, help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
# parser.add_argument('--sync_bn', action='store_true',default=True,help='enabling apex sync BN.') # 默认为False
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)


num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
print("the number of num_gpus is......", num_gpus,"\n")
is_distributed = num_gpus > 1
print(is_distributed,"\n")   
# is_distributed = False 

# print("is_get_rank....",dist.get_rank(),"\n")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用0号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"   # 总共0,1,2,3,4张卡，这里指定3就是使用3这张卡

# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, cpc loss = {:.3f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss,
                           scalar_outputs['depth_loss'],
                           scalar_outputs['cpc_loss'],
                           time.time() - start_time))
                    
                    # print(
                    #    "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                    #        epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                    #        optimizer.param_groups[0]["lr"], loss,
                    #        scalar_outputs['depth_loss'],
                    #        time.time() - start_time))
                del scalar_outputs, image_outputs

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    # 'model': model.module.state_dict(), 多GPU模型保存
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)
                        # print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                        #                                                     epoch_idx, args.epochs,
                        #                                                     batch_idx,
                        #                                                     len(TestImgLoader), loss,
                        #                                                     scalar_outputs["depth_loss"],
                        #                                                     time.time() - start_time))
                        
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, cpc loss = {:.3f}, time = {:3f}".format(
                                                                            epoch_idx, args.epochs,
                                                                            batch_idx,
                                                                            len(TestImgLoader), loss,
                                                                            scalar_outputs["depth_loss"],
                                                                            scalar_outputs["cpc_loss"],
                                                                            time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()
    # sample的数据类型，都是<class 'torch.Tensor'>

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    # imgs和cam_para是在增加图像合成损失时加入的
    imgs = sample_cuda["imgs"]
    cam_para = sample_cuda["proj_matrices"]
    # imgs_test = sample_cuda["imgs_test"]  # 测试代码
    # if imgs.equal(imgs_test):
    #     print("****** imgs as the same as imgs_test")
    #     print("***** the shape of imgs is: {}".format(imgs.shape))  # torch.Size([4, 5, 3, 512, 640])
    #     print("***** the shape of imgs_test is: {}".format(imgs_test.shape))  # torch.Size([4, 5, 3, 512, 640])

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    # 未加入图像合成损失
    # outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    outputs = model(imgs, cam_para, sample_cuda["depth_values"], sample_cuda["intrinsics_matrices"])
    # outputs = model(imgs, cam_para, sample_cuda["depth_values"],sample_cuda["intrinsics_matrices"])
    depth_est = outputs["depth"]

    # 未加入图像合成损失: total_loss、depth_loss
    # loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])
    # 加入图像合成损失： total_loss、depth_loss、total_cpc_loss
    loss, depth_loss, cpc_loss = model_loss(outputs, imgs, cam_para, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    # 2024.03.28, from transmvsnet: focal_loss + cross_view_loss 
    # loss, depth_loss, cpc_loss = model_loss(outputs, imgs, cam_para, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])
    
    if is_distributed and args.using_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()

    # scalar_outputs = {"loss": loss,
    #                   "depth_loss": depth_loss,
    #                   "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
    #                   "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
    #                   "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
    #                   "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),}
    
    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "cpc_loss": cpc_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),}

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    # imgs和cam_para是在增加图像合成损失时加入的
    imgs = sample_cuda["imgs"]
    cam_para = sample_cuda["proj_matrices"]
 

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    # outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    # outputs = model_eval(imgs, cam_para, sample_cuda["depth_values"], sample_cuda["intrinsics_matrices"])
    outputs = model_eval(imgs, cam_para, sample_cuda["depth_values"],sample_cuda["intrinsics_matrices"])
    depth_est = outputs["depth"]

    # 未加入图像合成损失
    # loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    # 加入图像合成损失： total_loss、depth_loss、total_cpc_loss
    loss, depth_loss, cpc_loss = model_loss(outputs, imgs, cam_para, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    # 2024.03.28, from transmvsnet: focal_loss + cross_view_loss 
    # loss, depth_loss, cpc_loss = model_loss(outputs, imgs, cam_para, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])
    
    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "cpc_loss": cpc_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
                      "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),

                      "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
                      "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [2.0, 4.0]),
                      "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [4.0, 8.0]),
                      "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [8.0, 14.0]),
                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [14.0, 20.0]),
                      "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [20.0, 1e5]),
                    }

    # scalar_outputs = {"loss": loss,
    #                   "depth_loss": depth_loss,
    #                   "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
    #                   "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
    #                   "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
    #                   "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
    #                   "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
    #                   "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),

    #                   "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
    #                   "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [2.0, 4.0]),
    #                   "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [4.0, 8.0]),
    #                   "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [8.0, 14.0]),
    #                   "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [14.0, 20.0]),
    #                   "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [20.0, 1e5]),
    #                 }

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)

# def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()
    # print("is use the sync_bn of apex .....",args.sync_bn,"\n")
    # print("is use the using_apex.....",args.using_apex,"\n")
    # print("is use the resume....",args.resume,"\n")
    # print("is distributed....",is_distributed,"\n")

    # using sync_bn by using nvidia-apex, need to install apex.
    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
        # print("using is_distributed......\n")

    set_random_seed(args.seed)
    device = torch.device(args.device)
    # print("device is.....", device, "\n")
    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method)
    model.to(device)
    model_loss = cas_mvsnet_loss


    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    # load parameters， from casmvsnet
    start_epoch = 0
    if args.resume:  # 自动从生成的模型文件夹中找
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

        
    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        # Initialize Amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            # print("Let's use.......", torch.cuda.device_count(), "GPUs!")
            print("the totall number of GPUS is.......", torch.cuda.device_count(), "GPUs!")
            # model = nn.DataParallel(model) # 也是属于多卡训练

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)  # dtu_yao.py/blendedmvs.py
    # train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.nviews, args.numdepth, args.interval_scale)  #12.26改为5
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.nviews, args.numdepth, args.interval_scale)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)
    else:
        # TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
        #                             pin_memory=args.pin_m)
        # TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
        #                            pin_memory=args.pin_m)  # 原始设置
        # 加入transformer后，UTL利用率低，将pip_memory改为true，num_workers改为4
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False,
                                   pin_memory=args.pin_m)
        # print("use the number of batchsize is......", args.batch_size,"\n")


    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    # elif args.mode == "profile":
    #     profile()
    else:
        raise NotImplementedError