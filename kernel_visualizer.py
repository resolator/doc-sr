#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Kernel Visualizer from KernelGAN."""
import argparse
import scipy.io as sio

from pathlib import Path
from matplotlib import pyplot as plt


def main():
    """Application entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-path', type=Path, required=True,
                        help='Path to kernel file.')
    args = parser.parse_args()
    
    img = sio.loadmat(args.kernel_path)['Kernel']
    plt.imsave(str(args.kernel_path) + '.png', img, cmap='gray')
    

if __name__ == '__main__':
    main()
