#!/bin/bash
rm -r ./cache_files
python train.py \
--name church_augment --batch 4 \
--dataroot_sketch ./data/sketch/photosketch/gabled_church \
--dataroot_image ./data/image/church --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-church/netG.pth \
--d_pretrained ./pretrained/stylegan2-church/netD.pth \
--max_iter 150000 --gan_mode ls --lr 1e-5 --name ganmode_ls-lr_1e-5 --disable_eval --diffaug_policy translation \
