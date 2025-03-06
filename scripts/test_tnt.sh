#!/usr/bin/env bash

# path to dataset
# TESTPATH = "/home/tianhaitao/repo/tankandtemples/intermediate"
# TESTPATH = "/root/repo/tht/tankandtemples/intermediate"
# TESTPATH="/home/tianhaitao/repo/tankandtemples/advanced" 

# TESTLIST = "./lists/tnt/intermediate.txt" 	
# TESTLIST="lists/tnt/advanced.txt" 										

# path to checkpoint
# CKPT_FILE="./checkpoints/test0401_depthN5_adapCost_cpc_64/model_000015.ckpt"   # 有TransMVSNet模块
# CKPT_FILE="./checkpoints/test0227_depthpriorN5_adapCost_64/model_000015.ckpt"  # 加入深度先验, cost volume adaptive, (64,32,8)	
# CKPT_FILE="./checkpoints/test0407_bldfinetune/model_000015.ckpt"  # 加入transmvsnet模块(使用自己loss)，并用blended进行finetune
CKPT_FILE="./checkpoints/test0412NoTrans_bldfinetune/model_000015.ckpt" # 没有加入Trans模块，使用的是自己原先模型，并用blend进行finetune

# path to save the results，  OUTDIR只是为了输出log	 
# LOG_DIR="./outputs_test0509NoTransFine_tnt_config/adv"  # 没加TransMVSNet任何模块，使用自己模块并用blend进行finetune
# LOG_DIR="./outputs_test20250226NoTransFine_tnt_config/adv"  #
LOG_DIR="./outputs_test20250228NoTransFine_tnt_config/inter"  #

if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi
# 一定要注意：这里的OUTDIR只是为了方便输出log，其和test.py里面的--outdir作用是不一样的
# --outdir的路径设置是为了模型一些输出文件内容

# python test.py --dataset=tnt_eval --batch_size=1 --interval_scale 1.06 \
#                --testpath $TESTPATH  --testlist $TESTLIST \  这种书写就提示报错
#                --loadckpt $CKPT_FILE \
#                --num_view=7 \
#                --filter_method 'dypcd' ${@:2} 

# python test.py --dataset=tnt_eval --batch_size=1 --interval_scale 1.06 \
#                --testpath "/home/tianhaitao/repo/tankandtemples/intermediate" \
#                --datapath "/home/tianhaitao/repo/tankandtemples/intermediate" \
#                --testlist "./lists/tnt/intermediate.txt"  \
#                --loadckpt $CKPT_FILE \
#                --num_view=7 \
#                --filter_method 'gipuma' 

# python test.py --dataset=general_eval --batch_size=1 --interval_scale 1.06 \
#                --testpath "/home/tianhaitao/repo/tankandtemples/intermediate"  \
#                --testlist "./lists/tnt/intermediate.txt"  \
#                --loadckpt $CKPT_FILE --fusibile_exe_path ./fusibile/build2/fusibile \
#                --filter_method 'gipuma' 

# intermediate:（1920/2048，1080）——> (1920/2048, 1056)
# 'gipuma融合'
# python test.py --dataset=tnt_eval_trans --batch_size=1 --interval_scale 1.06 \
#                --testpath "/home/tianhaitao/repo/tankandtemples/intermediate"  \
#                --testlist "./lists/tnt/intermediate.txt"  \
#                --loadckpt $CKPT_FILE --fusibile_exe_path ./fusibile/build2/fusibile \
#                --num_view=7 \
#                --ndepths "64,32,8" \
#                --prob_threshold=0.1 --disp_threshold=0.25 --num_consistent=3 \
#                --outdir "./outputs_test0421TransFine_tnt_gipuma/inter" \
#                --filter_method 'gipuma' ${@:2} 

# intermediate:（1920/2048，1080）——> (1920/2048, 1024)  采取geomvsnet的做法
# python test.py --dataset=tnt_eval_geo --batch_size=2 --interval_scale 1.06 \
#                --testpath "/home/tianhaitao/repo/tankandtemples/intermediate"  \
#                --testlist "./lists/tnt/intermediate.txt"  \
#                --loadckpt $CKPT_FILE --fusibile_exe_path ./fusibile/build2/fusibile \
#                --num_view=7 \
#                --ndepths "64,32,8" \
#                --outdir "./outputs_test0420TransFine_tnt1024/inter" \
#                --filter_method 'gipuma' ${@:2} 

# intermediate:（1920/2048，1080）——> (1920/2048, 1056)
# dypcd融合,  intermediate
# 第一次的config用的是RA-MVSNet的参数
# python test_uni.py --dataset=tnt_eval_trans --batch_size=2 --interval_scale 1.06 \
#                --testpath "/home/tianhaitao/database/tankandtemples/intermediate"  \
#                --datapath "/home/tianhaitao/database/tankandtemples/intermediate" \
#                --testlist "./lists/tnt/intermediate.txt"  \
#                --loadckpt $CKPT_FILE \
#                --num_view=11 \
#                --ndepths "64,32,8" \
#                --outdir "./outputs_test0509NoTransFine_tnt_config/inter" \
#                --filter_method "dypcd" ${@:2} | tee -a $LOG_DIR/log.txt

# dypcd融合,  advanced
python test_uni.py --dataset=tnt_eval_trans --batch_size=2 --interval_scale 1.06 \
               --testpath "/home/tianhaitao/database/tankandtemples/intermediate"  \
               --datapath "/home/tianhaitao/database/tankandtemples/intermediate" \
               --testlist "./lists/tnt/intermediate.txt"  \
               --loadckpt $CKPT_FILE \
               --num_view=11 \
               --ndepths "64,32,8" \
               --outdir "./outputs_test20250228NoTransFine_tnt_config/inter" \
               --filter_method "dypcd" ${@:2} | tee -a $LOG_DIR/log.txt