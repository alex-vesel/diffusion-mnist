import torch
import numpy as np

from configs.nn_config import NUM_TIMESTEPS
from utils.diffusion import get_alpha_t, get_alpha_bar_t, get_noise_level


def train_model(model, train_dataloader, optimizer, loss_fn, num_epochs, experiment_logger, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            for key in batch:
                batch[key] = batch[key].to(device)

            outputs = model(batch['noised_img'], batch['timestep'])
            loss = loss_fn(outputs, batch['noise'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        experiment_logger.log_scalar('train_loss', avg_loss, epoch)
        experiment_logger.save_model(model, optimizer, 'model', epoch)

        # generate 4 sample images
        model.eval()
        sample_img = torch.tensor(np.random.normal(0, 1, (4, 1, 32, 32)), dtype=torch.float32).to(device)
        sample_img = torch.clip(sample_img, -1, 1)
        for i in range(NUM_TIMESTEPS):
            t = NUM_TIMESTEPS - i
            alpha_t = get_alpha_t(t)
            alpha_bar_t = get_alpha_bar_t(t)
            multiplier = (1 - alpha_t) / (np.sqrt(1 - alpha_bar_t))
            sigma_t = np.sqrt((1 - get_alpha_bar_t(t-1)) / (1 - alpha_bar_t) * get_noise_level(t))

            z = torch.tensor(np.random.normal(0, 1, (4, 1, 32, 32)), dtype=torch.float32).to(device)
            if t == 1:
                z *= 0.0

            with torch.no_grad():
                error = model(sample_img, torch.tensor([t] * 4, dtype=torch.float32).to(device))
                sample_img = (sample_img - multiplier * error) / np.sqrt(alpha_t) + sigma_t * z
        sample_img = torch.clip(sample_img, -1, 1)

        sample_img = sample_img.cpu().numpy()
        for i in range(4):
            experiment_logger.log_image(f'sample_img_{i}', sample_img[i], epoch)


            