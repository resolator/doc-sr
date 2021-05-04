#!/bin/bash
mkdir demo/image demo/text demo/result
python3 demo_doc_gen.py --save-to ./demo/image
python3 predict_tesseract.py --images-dir ./demo/image --save-to ./demo/result
python3 accuracy.py --label-dir ./demo/text/ --tesseract-dir ./demo/result/