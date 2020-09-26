CUDA_VISIBLE_DEVICES=1 python ../eval.py \
                    --val_img_dir='../dataset/WFLW_test/images/' \
                    --val_landmarks_dir='../dataset/WFLW_test/landmarks/' \
                    --ckpt_save_path='../experiments/eval_iccv_0620' \
                    --hg_blocks=4 \
                    --pretrained_weights='../ckpt/WFLW_4HG.pth' \
                    --num_landmarks=98 \
                    --end_relu='False' \
                    --batch_size=20 \

