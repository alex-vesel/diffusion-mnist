import torch
import numpy as np
import matplotlib.pyplot as plt

from models.unet import SimpleUnet
from configs.nn_config import NUM_TIMESTEPS
from utils.diffusion import get_alpha_t, get_alpha_bar_t, get_noise_level


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else device)


model = SimpleUnet().to(device)
model_path = '/Users/alexvesel/Documents/diffusion-demo/experiments/test_2025-06-01_14-33-10/models/model_53.pth'
model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])


model.eval()
sample_img = torch.tensor(np.random.normal(0, 1, (1, 1, 32, 32)), dtype=torch.float32).to(device)
sample_img = torch.clip(sample_img, -1, 1)
intermediate_imgs = [sample_img.cpu().numpy()]
for i in range(NUM_TIMESTEPS):
    t = NUM_TIMESTEPS - i
    alpha_t = get_alpha_t(t)
    alpha_bar_t = get_alpha_bar_t(t)
    multiplier = (1 - alpha_t) / (np.sqrt(1 - alpha_bar_t))
    sigma_t = np.sqrt((1 - get_alpha_bar_t(t-1)) / (1 - alpha_bar_t) * get_noise_level(t))

    z = torch.tensor(np.random.normal(0, 1, (1, 1, 32, 32)), dtype=torch.float32).to(device)
    if t == 1:
        z *= 0.0

    with torch.no_grad():
        error = model(sample_img, torch.tensor([t] * 1, dtype=torch.float32).to(device))
        sample_img = (sample_img - multiplier * error) / np.sqrt(alpha_t) + sigma_t * z
    sample_img = torch.clip(sample_img, -1, 1)

    intermediate_imgs.append(sample_img.cpu().numpy())

# plot every 100th image in one figure, there is a total of NUM_TIMESTEPS + 1 images
intermediate_imgs = np.array(intermediate_imgs)
plt.figure(figsize=(20, 10))
for i in range(0, NUM_TIMESTEPS + 1, 100):
    plt.subplot(3, 5, i // 100 + 1)
    plt.imshow(intermediate_imgs[i][0][0], cmap='gray')
    plt.axis('off')
    plt.title(f'Timestep {i}')
plt.tight_layout()
plt.show()