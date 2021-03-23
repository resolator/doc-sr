#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Demonstrate the document generator."""
import cv2
import argparse
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

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    for i in range(args.img_count):
        img, text = gen_page(dpi=args.dpi,
                             mean_word_len=args.mean_word_len,
                             word_chars=args.word_chars)
        print(text)
        cv2.imshow('img', img)
        cv2.waitKey()


if __name__ == '__main__':
    main()
