import numpy as np
import os
import multiprocessing
import torch
from torchvision.transforms import Compose
from torch.utils.data import Dataset
import sys
sys.path.append(".")

from utils.custom_transforms import *
from configs.path_config import DATA_DIR
from configs.nn_config import NUM_TIMESTEPS, MIN_NOISE, MAX_NOISE
from utils.path_utils import recurse_dir_for_imgs
from utils.diffusion import get_alpha_t, get_alpha_bar_t, get_noise_level


class DataFile(Dataset):
    def __init__(self, clip_path, transform=None):
        self.clip_path = clip_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = cv2.imread(self.clip_path)
        img = img[..., [0]]
        img = np.transpose(img, (2, 0, 1))

        # upsample to 32x32
        img = cv2.resize(img[0], (32, 32), interpolation=cv2.INTER_LINEAR)
        img = np.expand_dims(img, axis=0)  # add channel dimension

        if self.transform:
            img = self.transform(img)

        # sample noise level
        t = np.random.randint(0, NUM_TIMESTEPS) + 1
        noised_img, noise = self.noise_img(img, t)

        out = {
            'noised_img': noised_img,
            'noise': noise,
            'timestep': t
        }

        return out

    def noise_img(self, img, t):
        alpha_bar_t = get_alpha_bar_t(t)
        noise = np.random.normal(0, 1, img.shape).astype(np.float32)
        noised_img = np.clip(np.sqrt(alpha_bar_t) * img + np.sqrt(1 - alpha_bar_t) * noise, -1, 1).astype(np.float32)
        return noised_img, noise


def aggregate_data(clip_paths, transform=None, num_workers=1):
    data = []
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            data = pool.starmap(DataFile, [(clip_path, transform) for clip_path in clip_paths])
    else:
        for clip_path in clip_paths:
            datafile = DataFile(clip_path, transform)
            data.append(datafile)

    return data


def split_data(datafiles, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0):
    np.random.shuffle(datafiles)
    n = len(datafiles)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = datafiles[:train_end]
    val_data = datafiles[train_end:val_end]
    test_data = datafiles[val_end:]

    return train_data, val_data, test_data


if __name__ == "__main__":
    img_paths = recurse_dir_for_imgs(DATA_DIR, file_extension='jpg')
    print(f"Found {len(img_paths)} directories with images.")

    transform = Compose([
        ToFloat(),
        NormalizeToRange(0, 255, -1, 1),
    ])

    datafiles = aggregate_data(img_paths, transform=transform)

    datafiles[0][0]