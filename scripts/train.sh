#!/usr/bin/env bash
# MVS_TRAINING="./data/DTU/mvs_training/dtu/"
MVS_TRAINING="/home/tianhaitao/database/DTU/mvs_training/"

# LOG_DIR="./checkpoints/debug/"
# LOG_DIR="./checkpoints/debug_cpcloss_1117/"
# LOG_DIR="./checkpoints/debug_test1222_depthprior/"  # N=3
# LOG_DIR="./checkpoints/debug_test1226_depthprior_N5/"  # N=5
# LOG_DIR="./checkpoints/debug_test1229_depthpriorN3_adapCost/"  # N=3,且cost volume聚合方式改为自适应权重
# LOG_DIR="./checkpoints/debug_test1231_depthpriorN5_adapCost"  # N=5,且cost volume聚合方式改为自适应权重
# LOG_DIR="./checkpoints/test0114_depthpriorN5_adapCost_edge"  # N=5,且cost volume adaptive, 加入edge_feature

# LOG_DIR="./checkpoints/test0128_depthpriorN5_adapCost_64"  # N=5,且cost volume adaptive(AA-RMVSNet),(64,32,8)
# LOG_DIR="./checkpoints/test0301_depthpriorN5_adapCost_64"  # N=5,且cost volume adaptive(Uni-MVSNet),(64,32,8)
# LOG_DIR="./checkpoints/test0312_depthN5_adapCost_nocpc_64"  # N=5,且cost volume adaptive(Uni-MVSNet),(64,32,8), 没有加入图像合成损失
# LOG_DIR="./checkpoints/test0318_depthN5_adapCost_nocpc_64"  # N=5,且cost volume adaptive(Uni-MVSNet),(64,32,8), 图像合成损失权重由12改为120
# LOG_DIR="./checkpoints/test0322_depthN5_adapCost_cpc_64"   # N=5,且cost volume adaptive(Uni-MVSNet),(64,32,8), 图像合成损失权重由12改为200
# LOG_DIR="./checkpoints/test0325_depthN5_adapCost_cpc_64"  # N=5,且cost volume adaptive(Uni-MVSNet),(64,32,8)，图像合成损失权重为1
# LOG_DIR="./checkpoints/test0328_depthN5_adapCost_cpc_64"  # N=5,且cost volume adaptive(Uni-MVSNet),(64,32,8)，引入TransMVSNet
LOG_DIR="./checkpoints/test0401_depthN5_adapCost_cpc_64"  # N=5,且cost volume adaptive(Uni-MVSNet),(64,32,8)，引入TransMVSNet,但损失函数依旧用L1
# LOG_DIR=$2
# LOG_DIR=$1
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

# 旧方式
# python -m torch.distributed.launch --nproc_per_node=$1 train.py --logdir $LOG_DIR --dataset=dtu_yao --batch_size=1 --trainpath=$MVS_TRAINING \
#                 --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt

#新方式
torchrun --nproc_per_node=$1 --master_port=29501 train.py --logdir $LOG_DIR \
         --dataset=dtu_yao --batch_size=4 --nviews=5 --trainpath=$MVS_TRAINING \
         --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt \
         --numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt  

# python  train.py --logdir $LOG_DIR --dataset=dtu_yao --trainpath=$MVS_TRAINING \
#                 --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt                            