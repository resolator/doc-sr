import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image

from kernelgan.util import create_gradient_map, create_probability_map, nn_interpolation
from imresize import imresize


class CropGenerator(Dataset):
    def __init__(self, img_path, d_input_shape, d_forward_shave, max_iters=3000, scale_factor=0.5, input_crop_size=64):
        # Default shapes
        self.scale_factor = scale_factor
        self.input_crop_size = input_crop_size
        self.max_iters = max_iters
        
        self.g_input_shape = self.input_crop_size
        self.d_input_shape = d_input_shape  # shape entering D downscaled by G
        self.d_output_shape = self.d_input_shape - d_forward_shave

        # Read and Preproccess image
        self.image = np.array(Image.open(img_path.absolute()).convert('RGB'), dtype=np.uint8) / 255.
        # self.image = self.image[10:-10, 10:-10, :]  # Crop pixels to avoid boundaries effects in synthetic examples
        # Crop pixels for the shape to be divisible by the scale factor
        sf = int(1 / self.scale_factor)
        self.image = self.image[:-(self.image.shape[0] % sf), :, :] if self.image.shape[0] % sf > 0 else self.image
        self.image = self.image[:, :-(self.image.shape[1] % sf), :] if self.image.shape[1] % sf > 0 else self.image

        self.in_rows, self.in_cols = self.image.shape[0:2]

        # Create prob map for choosing the crop
        prob_map_big, prob_map_sml = self.create_prob_maps()
        self.crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=self.max_iters, p=prob_map_sml)
        self.crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=self.max_iters, p=prob_map_big)

    def __len__(self):
        return self.max_iters

    def __getitem__(self, idx):
        """Get a crop for both G and D """
        g_in = self.next_crop(for_g=True)
        d_in = self.next_crop(for_g=False)

        return g_in, d_in

    def next_crop(self, for_g):
        """Return a crop according to the pre-determined list of indices. Noise is added to crops for D"""
        idx = np.random.randint(self.max_iters)
        
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left(size, for_g, idx)
        crop_im = self.image[top:top + size, left:left + size, :]
        if not for_g:
            crop_im += np.random.randn(*crop_im.shape) / 255.0  # Add noise to the image for d
        return torch.FloatTensor(np.transpose(crop_im, (2, 0, 1)) * 2.0 - 1.0).unsqueeze(0).cuda()

    def create_prob_maps(self):
        # Create loss maps for input image and downscaled one
        loss_map_big = create_gradient_map(self.image)
        loss_map_sml = create_gradient_map(imresize(im=self.image, scale_factor=self.scale_factor, kernel='cubic'))
        
        # Create corresponding probability maps
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / self.scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def get_top_left(self, size, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        center = self.crop_indices_for_g[idx] if for_g else self.crop_indices_for_d[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        
        # Choose even indices (to avoid misalignment with the loss map for_g)
        return top - top % 2, left - left % 2
    
    
class CropDataModule(pl.LightningDataModule):
    def __init__(self, img_path, max_iters, d_input_shape, d_forward_shave, bs=4):
        super().__init__()
        self.bs = bs
        self.data = CropGenerator(
            img_path=img_path,
            d_input_shape=d_input_shape,
            d_forward_shave=d_forward_shave,
            max_iters=max_iters
        )
        
    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.bs)
