#!/usr/bin/env bash
# TESTPATH="data/DTU/dtu_test_all"
TESTPATH="/home/tianhaitao/database/DTU/dtu_test"
# TESTLIST="lists/dtu/test.txt"   # 原始精度测试
TESTLIST="lists/dtu/test_dhps.txt"   # ADIA假设平面消融实验测试

# checkpoint file
# CKPT_FILE=$1
# CKPT_FILE="checkpoints/debug/model_000015.ckpt"

# path to checkpoint
# CKPT_FILE="./checkpoints/test0401_depthN5_adapCost_cpc_64/model_000015.ckpt"   # 有TransMVSNet模块
# CKPT_FILE="./checkpoints/test0227_depthpriorN5_adapCost_64/model_000015.ckpt"  # 加入深度先验, cost volume adaptive, (64,32,8)	
# CKPT_FILE="./checkpoints/test0407_bldfinetune/model_000015.ckpt"  # 加入transmvsnet模块(使用自己loss)，并用blended进行finetune
CKPT_FILE="./checkpoints/test0412NoTrans_bldfinetune/model_000015.ckpt" # 没有加入Trans模块，使用的是自己原先模型，并用blend进行finetune

# path to save the results，  OUTDIR只是为了输出log	 
# LOG_DIR="./outputs_test0509NoTransFine_tnt_config/adv"  # 没加TransMVSNet任何模块，使用自己模块并用blend进行finetune
# LOG_DIR="./outputs_test20250226NoTransFine_tnt_config/adv"  #
LOG_DIR="./outputs_test20250228NoTransFine_tnt_config/dtu"  #
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi

python test_uni.py --dataset=general_eval --batch_size=4 \
               --testpath=$TESTPATH  \
               --datapath=$TESTPATH \
               --testlist=$TESTLIST \
               --loadckpt $CKPT_FILE \
               --outdir "./outputs_test20250228NoTransFine_tnt_config/dtu" \
               --filter_method "dypcd" ${@:2} | tee -a $LOG_DIR/log.txt
