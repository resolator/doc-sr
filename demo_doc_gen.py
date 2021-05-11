#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Demonstrate the document generator."""
import cv2
import argparse

from tqdm import tqdm
from pathlib import Path
from doc_gen import gen_page


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img-count', type=int, default=5,
                        help='Number of images to generate.')
    parser.add_argument('--dpi', type=int, default=250,
                        help='DPI for generated images.')
    parser.add_argument('--mean-word-len', type=int, default=5,
                        help='Mean length of generated words.')
    parser.add_argument('--word-chars', action='store_true',
                        help='Generate only in-word characters.')
    parser.add_argument('--page-format', default='A5',
                        choices=['A3', 'A4', 'A5'],
                        help='Page format for generation.')
    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def main(args):
    """Application entry point."""
    if args.save_to is not None:
        args.save_to.mkdir(parents=True, exist_ok=True)

    else:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    for i in tqdm(range(args.img_count)):
        img, text = gen_page(dpi=args.dpi,
                             mean_word_len=args.mean_word_len,
                             word_chars=args.word_chars,
                             page_format=args.page_format)

        if args.save_to is not None:
            img_path = args.save_to.joinpath(str(i) + '.png')
            txt_path = args.save_to.joinpath(str(i) + '.txt')

            cv2.imwrite(str(img_path), img)
            with open(txt_path, 'w') as f:
                print(' '.join(text), file=f, end='')
        else:
            print(text)
            cv2.imshow('img', img)
            cv2.waitKey()


if __name__ == '__main__':
    args = get_args()
    main(args)
