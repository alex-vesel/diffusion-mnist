import numpy as np
from configs.nn_config import NUM_TIMESTEPS, MAX_NOISE, MIN_NOISE

def get_noise_level(t):
    return MIN_NOISE + (MAX_NOISE - MIN_NOISE) * (t / NUM_TIMESTEPS)

def get_alpha_t(t):
    return 1 - get_noise_level(t)

def get_alpha_bar_t(t):
    alpha_t = get_alpha_t(t)
    return np.prod([get_alpha_t(i) for i in range(1, t + 1)])