#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Script for predictions generation using KernelGan."""
import torch
import wandb
import argparse

import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pathlib import Path

from zssr.ZSSR import ZSSR
from kernelgan.dataloader import CropDataModule
from kernelgan.networks import KernelGAN


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Path to dir with images for SR.')
    parser.add_argument('--kg-max-iters', type=int, default=2750,
                        help='Iterations for KernelGAN.')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                        help='Noise scale for ZSSR.')
    parser.add_argument('--save-to', type=Path, required=True,
                        help='Path to save dir.')

    return parser.parse_args()


def train_kg(img_path, gan, max_iters=2750):
    data_dl = CropDataModule(
        img_path=img_path,
        d_input_shape=gan.d_input_shape,
        d_forward_shave=gan.D.forward_shave,
        max_iters=max_iters
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1
    )
    trainer.fit(gan, data_dl)
    wandb.finish()
    gan.post_process_kernel()


def main():
    """Application entry point."""
    args = get_args()
    args.save_to.mkdir(parents=True, exist_ok=True)

    for img_path in args.images_dir.glob('*.png'):
        gan = KernelGAN()
        train_kg(img_path, gan, args.kg_max_iters)
        kernel = gan.kernel

        # clean memory
        del gan
        torch.cuda.empty_cache()

        try:
            sr = ZSSR(
                img_path.absolute(),
                scale_factor=2,
                kernels=[kernel],
                is_real_img=True,
                noise_scale=args.noise_scale
            ).run()
        except Exception:
            continue

        # save sr
        img_save_to = args.save_to.joinpath(img_path.name)
        plt.imsave(
            img_save_to,
            sr,
            vmin=0,
            vmax=255 if sr.dtype == 'uint8' else 1.,
            dpi=1
        )


if __name__ == '__main__':
    main()