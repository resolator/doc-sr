#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Caclulate CER between two folders with txt files."""
import fastwer
import argparse

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

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    cers, lds = [], []
    for pd_path in tqdm(list(args.preds_dir.glob('*.txt')), 'Evaluating'):
        try:
            with open(pd_path, 'r') as f:
                pd_text = f.read()

            gt_path = args.labels_dir.joinpath(pd_path.name)
            with open(gt_path, 'r') as f:
                gt_text = f.read()

            cer = fastwer.score([pd_text], [gt_text], char_level=True)
            ld = levenshtein_distance(pd_text, gt_text)
            cers.append(cer)
            lds.append(ld)
        except Exception:
            continue

    print('Mean CER:', sum(cers) / len(cers))
    print('Mean Levenshtein distance:', sum(lds) / len(lds))


if __name__ == '__main__':
    main()