import os
import datetime
import torch
from tensorboardX import SummaryWriter

class ExperimentLogger:
    def __init__(self, logdir='experiments', exp_name="test"):
        self.logdir = logdir
        self.exp_name = exp_name
        self.create_exp_path()
        self.writer = SummaryWriter(self.exp_path)

    def create_exp_path(self):
        self.exp_path = os.path.join(self.logdir, self.exp_name)
        self.exp_path += datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.exp_path, exist_ok=True)
        self.model_path = os.path.join(self.exp_path, 'models')
        os.makedirs(self.model_path, exist_ok=True)

    def log_scalar(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def log_image(self, key, image, step):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        self.writer.add_image(key, image, step)

    def save_model(self, model, optimizer, name, step):
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{self.model_path}/{name}_{step}.pth')
