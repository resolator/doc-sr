#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
import fastwer
import argparse

import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-dir', type=Path, required=True,
                        help='Path to ground truth.')
    parser.add_argument('--lr-dir', type=Path, required=True,
                        help='Path to low resolution OCR predictions.')
    parser.add_argument('--sr-dir', type=Path, required=True,
                        help='Path to super resolution OCR predictions.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    lr_better, sr_better, total = 0, 0, 0
    deltas = []
    for sr_path in tqdm(list(args.sr_dir.glob('*.txt')), 'Evaluating'):
        # read data
        with open(sr_path, 'r') as f:
            sr_text = f.read()

        with open(args.lr_dir.joinpath(sr_path.name), 'r') as f:
            lr_text = f.read()

        with open(args.gt_dir.joinpath(sr_path.name), 'r') as f:
            gt_text = f.read()

        # calc cer
        lr_cer = fastwer.score([lr_text], [gt_text], char_level=True)
        sr_cer = fastwer.score([sr_text], [gt_text], char_level=True)

        # update score
        if lr_cer > sr_cer:
            sr_better += 1
            total += 1
        elif lr_cer < sr_cer:
            lr_better += 1
            total += 1

        deltas.append(sr_cer - lr_cer)

    print('SR better than LR:', sr_better)
    print('LR better than SR:', lr_better)
    print('SR / total:', sr_better / total)
    print('LR / total:', lr_better / total)

    plt.hist(deltas, bins=50)
    plt.grid(True)
    plt.title(args.sr_dir.name)
    plt.savefig('./' + args.sr_dir.name + '.png')


if __name__ == '__main__':
    main()
