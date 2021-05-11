# doc-sr
Project for documents super-resolution.

## Requirements

The code was tested on Ubuntu 20.04 with CUDA 11.0.

Install all python dependencies from `requirements.txt` file using (for 
example) the following command:
```sh
pip3 install --user -r requirements.txt
```


To allow prediction generation using tesseract install `tesseract-ocr` package 
using the following command:
```sh
sudo apt update && sudo apt install tesseract-ocr poppler-utils python3-opencv
```


## Usage example

```sh
./demo_doc_gen.py --img-count 300 --dpi 200 --word-chars
--page-format A5 --save-to ./dataset_200dpi
./predict_kernelgan.py --images-dir ./dataset_200dpi/ --save-to ./sr
```