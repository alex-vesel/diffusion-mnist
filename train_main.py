import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose
from utils.path_utils import recurse_dir_for_imgs

from utils.datafile import aggregate_data, split_data
from utils.custom_transforms import *
from utils.train import train_model
from models.unet import SimpleUnet
from utils.experiment_logger import ExperimentLogger

from configs.path_config import *
from configs.nn_config import *

np.random.seed(0)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else device)

def main():
    # get data
    img_paths = recurse_dir_for_imgs(DATA_DIR, file_extension='jpg')
    print(f"Found {len(img_paths)} image files.")

    transform = Compose([
        ToFloat(),
        NormalizeToRange(0, 255, -1, 1),
    ])

    datafiles = aggregate_data(img_paths, transform=transform)

    # create dataloaders
    train_dataset = ConcatDataset(datafiles)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    print(f"Created DataLoader with {len(train_dataset)} samples.")

    model = SimpleUnet().to(device)
    print("Number of parameters in the model:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # define loss function
    loss_fn = torch.nn.MSELoss()

    # create experiment logger
    experiment_logger = ExperimentLogger(logdir=EXPERIMENT_DIR)
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
        experiment_logger=experiment_logger,
        device=device
    )


if __name__ == "__main__":
    main()