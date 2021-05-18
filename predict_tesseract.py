#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Generate OCR predictions using tesseract."""
import argparse
import subprocess

from tqdm import tqdm
from pathlib import Path


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Path to directory with images.')
    parser.add_argument('--save-to', type=Path, required=True,
                        help='Path to save dir.')

    return parser.parse_args()


def ocr(img_path, remove_newline=False):
    """Run tesseract OCR on the given image and return recognized text.

    Parameters
    ----------
    img_path : str or Pathlib.Path
        Path to image for OCR.
    remove_newline : bool
        Remove newlines from output.

    Returns
    -------
    str
        Recognized text.

    """
    cmd = [
        '/usr/bin/tesseract',
        '-l', 'eng',
        '-c', 'tessedit_create_pdf=0',
        '-c', 'textonly_pdf=1',
        '-c', 'tessedit_create_txt=0',
        '-c', 'tessedit_pageseg_mode=1',
        str(img_path),
        'stdout'
    ]

    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    p.wait()
    output, error = p.communicate()
    text = output.decode('utf-8')

    if remove_newline:
        text = text.replace('\n', '')

    return text


def main():
    """Application entry point."""
    args = get_args()

    args.save_to.mkdir(parents=True, exist_ok=True)
    images_list = list(args.images_dir.glob('*.png'))
    for img_path in tqdm(images_list):
        text = ocr(img_path)

        # save recognized
        save_to = args.save_to.joinpath(img_path.stem + '.txt')
        with open(save_to, 'w') as f:
            print(text, file=f, end='')


if __name__ == '__main__':
    main()
