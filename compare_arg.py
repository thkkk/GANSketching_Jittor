import os
import argparse

import random
import numpy as np
import torch
from torchvision import utils
from training.networks.stylegan2 import Generator
import jittor as jt

g1 = Generator(256, 512, 8)
checkpoint = jt.load("checkpoint/teaser_cat_augment/12500_net_G.jt")
g1.load_state_dict(checkpoint)

g2 = Generator(256, 512, 8)
checkpoint = jt.load("pretrained/stylegan2-cat/netG.pth")
g2.load_state_dict(checkpoint)

print("--------style----------")
for p in g1.style.parameters():
    print(p)
    break

for p in g2.style.parameters():
    print(p)
    break

print("---------latent---------")
for p in g1.convs[2].parameters():
    print(p)
    break

for p in g2.convs[2].parameters():
    print(p)
    break