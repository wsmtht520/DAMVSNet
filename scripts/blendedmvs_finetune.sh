#!/usr/bin/env bash
MVS_TRAINING="/home/tianhaitao/database/dataset_low_res/"

# dtu checkpoint path
# 像unimvsnet和geomvsnet都采取的是将两个数据集混合起来进行测试
# resume = "./checkpoints/test0401_depthN5_adapCost_cpc_64/model_000015.ckpt"

# Transmvsnet采取的是先训练完一个，再训练另一个的方式
# CKPT="./checkpoints/test0401_depthN5_adapCost_cpc_64/model_000015.ckpt" # 基于加入Trans模块后
CKPT="./checkpoints/test0227_depthpriorN5_adapCost_64/model_000015.ckpt"

# LOG_DIR="./checkpoints/test0407_bldfinetune"   # 基于加入Trans模块后(使用的是自己loss)，又使用blend进行finetune
LOG_DIR="./checkpoints/test0412NoTrans_bldfinetune"  # 没有加入Trans模块，使用的是自己原先模块，又使用blend进行finetune
# LOG_DIR=$1
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

# dataset只是为了方便找到blendedmvs.py或dtu_yao.py
#新方式
torchrun --nproc_per_node=$1 --master_port=29501 finetune.py --logdir $LOG_DIR \
         --batch_size=2 \
         --nviews=7 \
         --trainpath $MVS_TRAINING \
         --loadckpt $CKPT \
         --dataset "blendedmvs" \
         --trainlist "lists/blendedmvs/training_list.txt" \
         --testlist "lists/blendedmvs/validation_list.txt" \
         --numdepth=128 ${@:3} | tee -a $LOG_DIR/log.txt  

