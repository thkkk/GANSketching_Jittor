import random
import pathlib
from PIL import Image
import jittor as jt
import numpy as np
from jittor.dataset import Dataset
from jittor.transform import ImageNormalize, Resize, Compose


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(Dataset):
    def __init__(self, path, image_mode='L', transform=None, max_images=None):
        super().__init__()
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        if max_images is None:
            self.files = files
        elif max_images < len(files):
            self.files = random.sample(files, max_images)
        else:
            print(f"max_images larger or equal to total number of files, use {len(files)} images instead.")
            self.files = files
        self.transform = transform
        self.image_mode = image_mode

    def __getitem__(self, index):
        image_path = self.files[index]
        image = Image.open(image_path).convert(self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.files)


def data_sampler(dataset, shuffle):
    if shuffle:
        return jt.dataset.RandomSampler(dataset)
    else:
        return jt.dataset.SequentialSampler(dataset)


def img_to_01_np(image: Image.Image):
    v = np.array(image, dtype=np.float32)
    if len(v.shape) == 2:
        v = v.reshape(1, v.shape[0], v.shape[1])
    else:
        v = v.transpose((2, 0, 1))
    v /= 255 # go to 0-1
    return v


def create_dataloader(data_dir, size, batch, img_channel=3):
    mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]
    transform = Compose([
        # 1. Resize
        # 2. normalize
        # 3. to var
        Resize(size),
        img_to_01_np,
        ImageNormalize(mean, std),
        jt.Var
    ])

    if img_channel == 1:
        image_mode = 'L'
    elif img_channel == 3:
        image_mode = 'RGB'
    else:
        raise ValueError("image channel should be 1 or 3, but got ", img_channel)

    dataset = ImagePathDataset(data_dir, image_mode, transform)

    sampler = data_sampler(dataset, shuffle=True)
    loader = dataset.set_attrs(batch_size=batch, sampler=sampler, drop_last=True)
    return loader, sampler


def yield_data(loader, sampler, distributed=False):
    epoch = 0
    while True:
        if distributed:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
