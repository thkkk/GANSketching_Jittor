#!/bin/bash
rm -r cache_files
python train.py \
--name standing_cat_augment --batch 4 \
--dataroot_sketch ./data/sketch/photosketch/standing_cat \
--dataroot_image ./data/image/cat --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-cat/netG.pth \
--d_pretrained ./pretrained/stylegan2-cat/netD.pth \
--max_iter 150000 --disable_eval --diffaug_policy translation \
