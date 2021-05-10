#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""OCR accuracy."""

import argparse
from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm
import os
from Levenshtein import distance as levenshtein_distance

def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-dir', type=Path, required=True,
                        help='Path to directory with labels.')
    parser.add_argument('--tesseract-dir', type=Path, required=True,
                        help='Path to directory with ocr.')

    return parser.parse_args()
    
def main():
    """Application entry point."""
    args = get_args()

    images_list = list(args.label_dir.glob('*.*'))
    for img_path in tqdm(images_list):
        head, tail = os.path.split(img_path)
        with open(img_path, "r") as f:
            content = f.read()
            content_list = content.split(",")
        
        with open(os.path.join(args.tesseract_dir, tail), "r") as f:
            ocr_content = f.read()
            ocr_list = ocr_content.split(",")
        print(levenshtein_distance(ocr_content, content))
        


if __name__ == '__main__':
    main()