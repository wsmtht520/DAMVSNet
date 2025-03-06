# EG-MVSNet


## ⚙ Setup

#### 1. Recommended environment

- PyTorch 1.12
- Python 3.8

#### 2. DTU Dataset

**Training Data**. We adopt the full resolution ground-truth depth provided in CasMVSNet or MVSNet. Download [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). 
Unzip them and put the `Depth_raw` to `dtu_training` folder. The structure is just like:

```
dtu_training                          
       ├── Cameras                
       ├── Depths   
       ├── Depths_raw
       └── Rectified
```

**Testing Data**. Download [DTU testing data](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) and unzip it. The structure is just like:

```
dtu_testing                          
       ├── Cameras                
       ├── scan1   
       ├── scan2
       ├── ...
```

#### 3. BlendedMVS Dataset

**Training Data** and **Validation Data**. Download [BlendedMVS](https://drive.google.com/file/d/1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb/view) and 
unzip it. And we only adopt 
BlendedMVS for finetuning and not testing on it. The structure is just like:

```
blendedmvs                          
       ├── 5a0271884e62597cdee0d0eb                
       ├── 5a3ca9cb270f0e3f14d0eddb   
       ├── ...
       ├── training_list.txt
       ├── ...
```

#### 4. Tanks and Temples Dataset

**Testing Data**. Download [Tanks and Temples](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) and 
unzip it. Here, we adopt the camera parameters of short depth range version (Included in your download), therefore, you should 
replace the `cams` folder in `intermediate` folder with the short depth range version manually. The 
structure is just like:

```
tanksandtemples                          
       ├── advanced                 
       │   ├── Auditorium       
       │   ├── ...  
       └── intermediate
           ├── Family       
           ├── ...          
```

## 📊 Testing

#### 1. DTU testing

**Fusibile installation**. Since we adopt Gipuma to filter and fuse the point on DTU dataset, you need to install 
Fusibile first. Download [fusible](https://github.com/YoYo000/fusibile) to `<your fusibile path>` and execute the following commands:

```
cd <your fusibile path>
cmake .
make
```

**Point generation**. To recreate the results from our paper, you need to specify the `datapath` to 
`<your dtu_testing path>`, `outdir` to `<your output save path>`, `resume` 
 to `<your model path>`, and `fusibile_exe_path` to `<your fusibile path>/fusibile` in shell file `./script/test.sh` first and then run:

```
bash ./scripts/test.sh
```



#### 2. Tanks and Temples testing

**Point generation**. Similarly, you need specify the `datapath`, `outdir` and `resume` in shell file 
`./scripts/test_tnt.sh`, and then run:

```
bash ./scripts/test_tnt.sh
```



## ⏳ Training

#### 1. DTU training

To train the model from scratch on DTU, specify the `datapath` and `log_dir` 
in `./scripts/train.sh` first 
and then run:

```
bash ./scripts/train.sh
```



#### 2. BlendedMVS fine-tuning

To fine-tune the model on BlendedMVS, you need specify `datapath`, `log_dir` and
`resume` in `./scripts/blendedmvs_finetune.sh` first, then run:

```
bash ./scripts/blendedmvs_finetune.sh
```



