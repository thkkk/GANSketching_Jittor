import os
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
from cleanfid import fid

from eval.ppl import compute_ppl
from training.networks.stylegan2 import Generator
import jittor as jt


def save_image(img, name):
    """Helper function to save jittor Var into an image file."""
    jt.misc.save_image(
        (img + 1) / 2,
        name,
        nrow=1,
        padding=0
    )


def make_eval_images(g, save_folder, eval_samples, batch_size):
    print(f"make eval imgs, eval_samples = {eval_samples}, batch_size = {batch_size}")
    if not os.path.exists(f'{save_folder}/image/'):
        os.makedirs(f'{save_folder}/image/')

    iterations = int(np.ceil(eval_samples / batch_size))
    images_left = eval_samples
    img_count = 0
    trunc_psi = 0.5
    mean_latent = g.mean_latent(8192)
    for i in range(iterations):
        batch = min(batch_size, images_left)
        images_left -= batch_size
        noise = jt.randn(batch, 512)
        sample, _ = g([noise], truncation=trunc_psi, truncation_latent=mean_latent)

        for ind in range(sample.size(0)):
            save_image(sample[ind], f'{save_folder}/image/{str(img_count).zfill(6)}.png')
            img_count += 1

def get_metrics(opt, name, target):
    real_folder = f"{opt.eval_root}/{target}/"
    fake_folder = f"{opt.sample_root}/{name}/"

    ckpt_path = f"{opt.model_root}/{name}.jt"
    g = setup_generator(ckpt_path)
    stats_fake = get_stats(opt, g, fake_folder)
    stats_real = get_stats(opt, None, real_folder)
    fid_value = fid.compute_fid(real_folder+'image', fake_folder+'image', num_workers=0)
    # ppl_wend = compute_ppl(g, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=10)
    del g

    fake_feats, real_feats = stats_fake['vgg_features'], stats_real['vgg_features']
    precision, recall = run_precision_recall(real_feats, fake_feats)

    return {
        "fid": fid_value,
        # "ppl": ppl_wend,
        "precision": precision,
        "recall": recall
    }


def get_stats(opt, g, folder):
    file_cached = False
    if os.path.exists(f'{folder}/image/'):
        if len([s for s in os.listdir(f'{folder}/image/') if s.endswith('.png')]) >= opt.eval_samples:
            file_cached = True

    if not file_cached:
        make_eval_images(g, folder, opt.eval_samples, opt.batch_size)

    vgg_features = get_vgg_features(folder, opt.eval_samples, opt.batch_size)
    return {
        "vgg_features": vgg_features
    }


def get_vgg_features(folder, eval_samples, batch_size):
    if os.path.exists(f'{folder}/vgg_features.npz'):
        f = np.load(f'{folder}/vgg_features.npz')
        features = f['feat']
        f.close()
        return features

    return run_vgg(folder, eval_samples, batch_size)


def run_vgg(folder, eval_samples, batch_size):
    from eval.precision_recall import metrics as pr
    pr.init_tf()
    # Initialize VGG-16.
    feature_net = pr.initialize_feature_extractor()

    # Calculate VGG-16 features.
    features = pr.get_features(f'{folder}/image/', feature_net, eval_samples, batch_size, num_gpus=1)
    np.savez_compressed(f"{folder}/vgg_features.npz", feat=features)

    return features


def run_precision_recall(real_feats, fake_feats):
    from eval.precision_recall import metrics as pr
    pr.init_tf()
    state = pr.knn_precision_recall_features(real_feats, fake_feats)
    precision = state['precision'][0]
    recall = state['recall'][0]
    return precision, recall


def setup_generator(ckpt_path, w_shift=False):
    g = Generator(256, 512, 8, w_shift=w_shift)
    ckpt = jt.load(ckpt_path)
    g.load_state_dict(ckpt)
    g.eval()
    return g


if __name__ == '__main__':
    """
    Example:
    python run_metrics.py --models_list weights/eval_list --model_root checkpoint --output metric_results.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_list', type=str)
    parser.add_argument('--output', type=str, default='metric_results.csv')
    parser.add_argument('--model_root', type=str, default='./weights')
    parser.add_argument('--eval_root', type=str, default='./data/eval/eval')
    parser.add_argument('--sample_root', type=str, default='./output')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--eval_samples', type=int, default=2500)
    opt = parser.parse_args()

    jt.flags.use_cuda = 1

    with open(opt.models_list, 'r') as f:
        lst = [s.strip().split(' ') for s in f.readlines()]
        all_models, all_targets = zip(*lst)

    with jt.no_grad():
        metrics = OrderedDict()
        for name, target in zip(all_models, all_targets):
            metrics[name] = get_metrics(opt, name, target)
            print(f"({name}) {metrics[name]}")

            table_columns = ['fid', 'ppl', 'precision', 'recall']
            table = pd.DataFrame.from_dict(metrics, orient='index', columns=table_columns)
            print("output:")
            table.to_csv(opt.output, na_rep='--')
