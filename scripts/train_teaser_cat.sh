#!/bin/bash
rm -r cache_files
python train.py \
--name teaser_cat --batch 4 \
--dataroot_sketch ./data/sketch/by_author/cat \
--dataroot_image ./data/image/cat --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-cat/netG.pth \
--d_pretrained ./pretrained/stylegan2-cat/netD.pth \
--disable_eval --diffaug_policy translation \
--display_freq 100 --save_freq 1000 --checkpoints_dir checkpoint --resume_iter 7000
