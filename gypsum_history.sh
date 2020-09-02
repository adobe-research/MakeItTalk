__='
   This is the default license template.
   
   File: gypsum_history.sh
   Author: dinli
   Copyright (c) 2020 dinli
   
   To edit this license information: Press Ctrl+Shift+P and press 'Create new License Template...'.
'

./sbatch_1080tilong.sh "--train --write --name" i2i_1gpu_1

./sbatch_fl_1080tilong.sh 0 100
./sbatch_fl_1080tilong.sh 100 200
./sbatch_fl_1080tilong.sh 200 300
./sbatch_fl_1080tilong.sh 300 400
./sbatch_fl_1080tilong.sh 400 500
./sbatch_fl_1080tilong.sh 500 600
./sbatch_fl_1080tilong.sh 600 700
./sbatch_fl_1080tilong.sh 700 800
./sbatch_fl_1080tilong.sh 800 900
./sbatch_fl_1080tilong.sh 900 1000


# multi-gpu version
./sbatch_1080tilong.sh "--train --write --batch_size 32 --name" i2i_2gpu_b32_1080ti 2 24G 1080ti-long
./sbatch_1080tilong.sh "--train --write --batch_size 32 --name" i2i_2gpu_b32_titanx 2 24G titanx-long
./sbatch_1080tilong.sh "--train --write --batch_size 32 --name" i2i_2gpu_b32_2080ti 2 24G 2080ti-long

./sbatch_1080tilong.sh "--train --write --batch_size 32 --jpg_freq 120 --ckpt_freq 720 --name" i2i_4gpu_b32_2080ti 4 24G 2080ti-long

./sbatch_1080tilong.sh "--train --write --batch_size 128 --name" i2i_8gpu_b128_1080ti 8 48G 1080ti-long
./sbatch_1080tilong.sh "--train --write --batch_size 64 --jpg_freq 120 --ckpt_freq 720 --name" i2i_8gpu_b64_1080ti 8 48G 1080ti-long
./sbatch_1080tilong.sh "--train --write --batch_size 32 --jpg_freq 120 --ckpt_freq 720 --name" i2i_8gpu_b32_2080ti 8 48G 2080ti-long

# train on preprocessed vox data
./sbatch_1080tilong.sh "--train --write --num_workers 8 --load_G_name /mnt/nfs/scratch1/yangzhou/VoxCeleb2_imagetranslation/ckpt/i2i_1gpu_1/ckpt_69.pth --batch_size 32 --jpg_freq 120 --ckpt_freq 720 --name" i2i_pre_load1gpu_4gpu_b32_1080ti 4 64G 1080ti-long
./sbatch_1080tilong.sh "--train --write --num_workers 16 --load_G_name /mnt/nfs/scratch1/yangzhou/VoxCeleb2_imagetranslation/ckpt/i2i_1gpu_1/ckpt_69.pth --batch_size 32 --jpg_freq 120 --ckpt_freq 720 --name" i2i_pre_load1gpu_4gpu_b32_2080ti 4 128G 2080ti-long
./sbatch_1080tilong.sh "--train --write --num_workers 8 --batch_size 32 --jpg_freq 120 --ckpt_freq 720 --name" i2i_pre_init_4gpu_b32_1080ti 4 64G 1080ti-long
./sbatch_1080tilong.sh "--train --write --num_workers 16 --batch_size 32 --jpg_freq 120 --ckpt_freq 720 --name" i2i_pre_init_4gpu_b32_2080ti_2 4 128G 2080ti-long

# 04/08 night train with style loss + minor (ckpt freq)
./sbatch_1080tilong.sh "--train --write --num_workers 16 --load_G_name /mnt/nfs/scratch1/yangzhou/VoxCeleb2_imagetranslation/ckpt/i2i_1gpu_1/ckpt_88.pth --batch_size 16 --name" i2istyle_rawcompress_load1gpu_4gpu_b16_2080ti 4 64G 2080ti-long
./sbatch_1080tilong.sh "--train --write --num_workers 16 --batch_size 16 --name" i2istyle_rawcompress_init_4gpu_b16_2080ti 4 64G 2080ti-long
./sbatch_1080tilong.sh "--train --write --num_workers 16 --load_G_name /mnt/nfs/scratch1/yangzhou/VoxCeleb2_imagetranslation/ckpt/i2i_1gpu_1/ckpt_88.pth --batch_size 16 --jpg_freq 120 --ckpt_last_freq 3600 --name" i2istyle_rawcompress_load1gpu_4gpu_b32_titanx 4 64G titanx-long
./sbatch_1080tilong.sh "--train --write --num_workers 16 --batch_size 16 --jpg_freq 120 --ckpt_last_freq 3600 --name" i2istyle_rawcompress_init_4gpu_b32_titanx 4 128G titanx-long
# -> on preprocessed as well
./sbatch_1080tilong.sh "--train --write --use_vox_dataset preprocessed --num_workers 16 --load_G_name /mnt/nfs/scratch1/yangzhou/VoxCeleb2_imagetranslation/ckpt/i2i_1gpu_1/ckpt_88.pth --batch_size 16 --name" i2istyle_preprocessed_load1gpu_4gpu_b16_1080ti 4 64G 1080ti-long
./sbatch_1080tilong.sh "--train --write --use_vox_dataset preprocessed --num_workers 16 --batch_size 16 --name" i2istyle_preprocessed_init_4gpu_b16_1080ti 4 64G 1080ti-long

# finetune on preprocessed (train / tune both on style loss)
./sbatch_1080tilong.sh "--train --write --use_vox_dataset preprocessed --num_workers 16 --load_G_name /mnt/nfs/scratch1/yangzhou/PreprocessedVox_imagetranslation/ckpt/i2istyle_rawcompress_init_4gpu_b16_2080ti/ckpt_35.pth --batch_size 16 --ckpt_epoch_freq 5 --name" i2itune_preprocessed_loadstyle_4gpu_b16_2080ti 4 64G 2080ti-long


# 04/28
# train on improved face alignment -> 3rdparty/AwingNet
srun -p 2080ti-short --gres=gpu:7 --mem=64G python main_train_image_translation.py --use_vox_dataset process --batch_size 112 --num_workers 7 --train --name awing_tmp_2
./sbatch_1080tilong.sh "--train --write --use_vox_dataset raw --num_workers 4 --batch_size 16 --ckpt_epoch_freq 1 --name" awing_1080 1 64G 1080ti-long
./sbatch_1080tilong.sh "--train --write --use_vox_dataset process --num_workers 4 --batch_size 16 --ckpt_epoch_freq 4 --name" awing_process_1080 1 64G 1080ti-long
srun -p m40-short --gres=gpu:1 --mem=32G python main_train_image_translation.py --use_vox_dataset raw --batch_size 1 --num_workers 0 --name test_awing --load_G_name /mnt/nfs/scratch1/yangzhou/PreprocessedVox_imagetranslation/ckpt/i2itune_preprocessed_loadstyle_4gpu_b16_2080ti/ckpt_150.pth

# debug vgg loss multi gpu
./sbatch_1080tilong.sh "--train --write --use_vox_dataset raw --num_workers 16 --batch_size 64 --ckpt_epoch_freq 1 --name" awing_raw_2080 4 64G 2080ti-long
./sbatch_1080tilong.sh "--train --write --use_vox_dataset process --num_workers 16 --batch_size 64 --ckpt_epoch_freq 5 --name" awing_process_2080 4 64G 2080ti-long

# 05/01
# add audio in and comb fan awing feature
srun -p 1080ti-short --gres=gpu:4 --mem=64G python main_train_image_translation.py --use_vox_dataset raw --batch_size 64 --num_workers 16 --train --name comb_tmp_1 --comb_fan_awing --test_speed
./sbatch_1080tilong.sh "--train --write --use_vox_dataset raw --num_workers 16 --batch_size 64 --ckpt_epoch_freq 1 --add_audio_in --name" audio_raw_1080 4 64G 1080ti-long
./sbatch_1080tilong.sh "--train --write --use_vox_dataset raw --num_workers 16 --batch_size 64 --ckpt_epoch_freq 1 --comb_fan_awing --name" comb_raw_1080 4 64G 1080ti-long






# test preprocessed vox
srun -p 2080ti-short --gres=gpu:1 --mem=32G python main_train_image_translation.py --num_workers 0 --load_G_name /mnt/nfs/scratch1/yangzhou/PreprocessedVox_imagetranslation/ckpt/awing_raw_2080/ckpt_47.pth --batch_size 1 --name test_tmp --use_vox_dataset raw
