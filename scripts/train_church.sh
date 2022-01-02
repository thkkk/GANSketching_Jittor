#!/bin/bash
rm -r ./cache_files
python train.py \
--name church_wo_aug --batch 4 \
--dataroot_sketch ./data/sketch/photosketch/gabled_church \
--dataroot_image ./data/image/church --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-church/netG.pth \
--d_pretrained ./pretrained/stylegan2-church/netD.pth \
--disable_eval --display_freq 200 --save_freq 1000 \
--resume_iter 9000 --checkpoints_dir checkpoint