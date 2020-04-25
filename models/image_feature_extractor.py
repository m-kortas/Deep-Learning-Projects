import os

import cv2
from dresses_utils import *
import numpy as np
import matplotlib.pyplot as plt

absolute_path = ""
FILE = absolute_path + "dresses\\train\\labels.txt"
VALIDATION_FILE = absolute_path +"dresses\\val\\labels.txt"

#większość sieci działa na kwadratowych obrazach

#skoro istotny jest kolor, to może dzielić obrazy na kwadratowy core i resztkowe oparte na histogramie (?)
#analiza histogramu barw

def main():

    n = read_number_of_lines(FILE);
    print(n)

    n=10
    #lines = read_data(FILE,"*")
    lines = read_data(FILE,n)

    dirname = os.path.dirname(FILE)
    images, labels = create_images_labels(lines,dirname)

    img = images[3]
    cv2.imshow('image', img)
    cv2.waitKey(0)

    W = np.zeros((n,1))
    H = np.zeros((n, 1))

    for i,img in enumerate(images):
        H[i], W[i] = img.shape[:2]

   # plt.hist(W)
   # plt.show()

   # plt.hist(H)
   # plt.show()

   # plt.hist(H/W)
   # plt.show()


    plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    #cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    plt.show()
    images = list(map(lambda img: rescale_image(img,(128,128)),images))

    img = images[3]
    cv2.imshow('image', img)
    cv2.waitKey(0)

    print(color_names)

    #images = np.array(images)
    #labels = np.array(labels)


if __name__ == "__main__":
    main()
