import torch
import numpy as np

def create_circular_mask(height, width, cutoff_ratio):
    """
    Create a circular mask for frequency domain filtering.
    """
    center_h, center_w = height // 2, width // 2
    radius = int(center_h, center_w)[0] * cutoff_ratio

    y, x = np.ogrid[:height, :width]
    mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= radius ** 2

    return torch.from_numpy(mask.astype(np.float32))

def create_rectangular_mask(height, width, cutoff_ratio):
    pass