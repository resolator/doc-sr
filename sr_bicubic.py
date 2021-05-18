#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Super resolution using bicubic interpolation."""
import cv2
import argparse

from tqdm import tqdm
from pathlib import Path


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Path to dir with images for SR.')
    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()
    args.save_to.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(args.images_dir.glob('*.png')):
        img = cv2.imread(str(img_path))
        upscaled = cv2.resize(img, None, 2, 2, cv2.INTER_CUBIC)
        img_save_to = args.save_to.joinpath(img_path.name)
        cv2.imwrite(str(img_save_to), upscaled)


if __name__ == '__main__':
    main()
