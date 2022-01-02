import argparse

import torch
import numpy as np
from tqdm import tqdm

import lpips
import jittor as jt
import jittor.nn as nn


def normalize(x):
    return x / jt.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * jt.acos(d)
    c = normalize(b - d * a)
    d = a * jt.cos(p) + c * jt.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


def compute_ppl(g, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=25, device='cuda'):
    percept = lpips.LPIPS(net='vgg').to(device)

    distances = []

    n_batch = num_samples // batch_size
    resid = num_samples - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid] if resid != 0 else [batch_size] * n_batch

    with jt.no_grad():
        with torch.no_grad():
            for batch in tqdm(batch_sizes):
                noise = g.make_noise()
                latent_dim = 512
                inputs = jt.randn([batch * 2, latent_dim])
                if sampling == 'full':
                    lerp_t = jt.rand(batch)
                else:
                    lerp_t = jt.zeros(batch)

                if space == 'w':
                    latent = g.get_latent(inputs)
                    latent_t0, latent_t1 = latent[::2], latent[1::2]
                    latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
                    latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + epsilon)
                    latent_e = jt.stack([latent_e0, latent_e1], 1).view(*latent.shape)

                image, _ = g([latent_e], input_is_latent=True, noise=noise)

                if crop:
                    c = image.shape[2] // 8
                    image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

                factor = image.shape[2] // 256

                if factor > 1:
                    image = nn.interpolate(
                        image, size=(256, 256), mode='bilinear', align_corners=False
                    )

                image = torch.tensor(image.numpy()).to(device)
                dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                    epsilon ** 2
                )
                distances.append(dist.cpu().numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    ppl = filtered_dist.mean()
    return ppl
