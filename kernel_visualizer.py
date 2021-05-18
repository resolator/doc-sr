import scipy.io as sio
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np

import argparse


def main():
    parser = argparse.ArgumentParser(description='Kernel Visualizer')
    parser.add_argument('kernel_path')
    args = parser.parse_args()
    
    img = sio.loadmat(args.kernel_path)['Kernel']
    plt.imsave(args.kernel_path+".png", img, cmap='gray')
    #plt.imshow(img, cmap='gray')
    #plt.axis('off')
    #plt.savefig(args.kernel_path+".png")
    

if __name__ == '__main__':
    main()
