## Sketch Your Own GAN

论文Sketch Your Own GAN([Wang et al.](https://github.com/PeterWang512/GANSketching))意在构建一个生成式模型GAN，这个模型可以通过一些简单的素描图，控制生成的彩图特征。也即，生成一些彩图，彩图的轮廓与输入的素描图大体相当。

我们使用[Jittor(计图)](https://github.com/Jittor/Jittor)来对该工作进行复现。

## Results


## Getting Started

### Clone our repo

```bash
git clone git@github.com:PeterWang512/GANSketching.git
cd GANSketching
```

### Install packages
- Install Jittor: Please refer to https://cg.cs.tsinghua.edu.cn/jittor/download/.
- Install other requirements:

  ```bash
  pip install -r requirements.txt
  ```

### Download model weights

- Run `bash weights/download_weights.sh`


### Generate samples from a customized model

This command runs the customized model specified by `ckpt`, and generates samples to `save_dir`.

```
# generates samples from the "standing cat" model.
python generate.py --ckpt weights/photosketch_standing_cat_noaug.pth --save_dir output/samples_standing_cat

# generates samples from the cat face model in Figure. 1 of the paper.
python generate.py --ckpt weights/by_author_cat_aug.pth --save_dir output/samples_teaser_cat

# generates samples from the customized ffhq model.
python generate.py --ckpt weights/by_author_face0_aug.pth --save_dir output/samples_ffhq_face0 --size 1024 --batch_size 4
```

### Latent space edits by GANSpace

Our model preserves the latent space editability of the original model. Our models can apply the same edits using the latents reported in Härkönen et.al. ([GANSpace](https://github.com/harskish/ganspace)).

```
# add fur to the standing cats
python ganspace.py --obj cat --comp_id 27 --scalar 50 --layers 2,4 --ckpt weights/photosketch_standing_cat_noaug.pth --save_dir output/ganspace_fur_standing_cat

# close the eyes of the standing cats
python ganspace.py --obj cat --comp_id 45 --scalar 60 --layers 5,7 --ckpt weights/photosketch_standing_cat_noaug.pth --save_dir output/ganspace_eye_standing_cat
```

## Model Training

Training and evaluating on model trained on PhotoSketch inputs requires running [the Precision and Recall metric](https://github.com/kynkaat/improved-precision-and-recall-metric). The following command pulls the submodule of the forked Precision and Recall [repo](https://github.com/PeterWang512/precision_recall).

```bash
git submodule update --init --recursive
```

### Download Datasets and Pre-trained Models

The following scripts downloads our sketch data, our evaluation set, [LSUN](https://dl.yf.io/lsun), and pre-trained models from [StyleGAN2](https://github.com/NVlabs/stylegan2) and [PhotoSketch](https://github.com/mtli/PhotoSketch).

```bash
# Download the sketches
bash data/download_sketch_data.sh

# Download evaluation set
bash data/download_eval_data.sh

# Download pretrained models from StyleGAN2 and PhotoSketch
bash pretrained/download_pretrained_models.sh

# Download LSUN cat, horse, and church dataset
bash data/download_lsun.sh
```

To train FFHQ models with image regularization, please download the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) using this [link](https://drive.google.com/file/d/1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv/view?usp=sharing). This is the zip file of 70,000 images at 1024x1024 resolution. Unzip the files, , rename the `images1024x1024` folder to `ffhq` and place it in `./data/image/`.


### Training Scripts

The example training configurations are specified using the scripts in `scripts` folder. Use the following commands to launch trainings.

```bash
# Train the "horse riders" model
bash scripts/train_photosketch_horse_riders.sh

# Train the cat face model in Figure. 1 of the paper.
bash scripts/train_teaser_cat.sh

# Train on a single quickdraw sketch
bash scripts/train_quickdraw_single_horse0.sh

# Train on sketches of faces (1024px)
bash scripts/train_authorsketch_ffhq0.sh
```

The training progress is tracked using `wandb` by default. To disable wandb logging, please add the `--no_wandb` tag to the training script.

### Evaluations
