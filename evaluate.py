#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Caclulate CER between two folders with txt files."""
import fastwer
import argparse

import numpy as np

from tqdm import tqdm
from pathlib import Path
from Levenshtein import distance as levenshtein_distance


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-dir', type=Path, required=True,
                        help='Path to directory with gt.')
    parser.add_argument('--preds-dir', type=Path, required=True,
                        help='Path to directory with ocr predictions.')
    parser.add_argument('--ref-dir', type=Path, required=True,
                        help='Path to dir with KernelGAN sr images.')
    parser.add_argument('--reject', type=float)

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    cers, lds = [], []
    for ref_path in tqdm(list(args.ref_dir.glob('*.png')), 'Evaluating'):
        pd_path = args.preds_dir.joinpath(ref_path.stem + '.txt')
        try:
            with open(pd_path, 'r') as f:
                pd_text = f.read()

            gt_path = args.labels_dir.joinpath(pd_path.name)
            with open(gt_path, 'r') as f:
                gt_text = f.read()

            cer = fastwer.score([pd_text], [gt_text], char_level=True)

            if args.reject is not None:
                if cer > args.reject:
                    continue

            ld = levenshtein_distance(pd_text, gt_text)
            cers.append(cer)
            lds.append(ld)
        except Exception:
            continue

    print('Total num:', len(cers))
    print('Mean CER:', sum(cers) / len(cers))
    print('Median CER:', np.median(cers))
    print('Mean Levenshtein distance:', sum(lds) / len(lds))


if __name__ == '__main__':
    main()