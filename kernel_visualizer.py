import scipy.io as sio
from matplotlib import pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description='Kernel Visualizer')
    parser.add_argument('kernel_path')
    args = parser.parse_args()
    
    img = sio.loadmat(args.kernel_path)['Kernel']
    plt.imsave(args.kernel_path+".png", img, cmap='gray')
    

if __name__ == '__main__':
    main()
