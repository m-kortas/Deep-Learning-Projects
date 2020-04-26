import os
import cv2
from dresses_utils import *

absolute_path = ""
FILE = absolute_path + "dresses\\train\\labels.txt"
VALIDATION_FILE = absolute_path +"dresses\\val\\labels.txt"


def main():
    lines = read_data(FILE,100)
    #print(lines)

    dirname = os.path.dirname(FILE)
    images, labels = create_images_labels(lines,dirname)

    print(labels)

    cv2.imshow('image', images[0])
    cv2.waitKey(0)

    #images = np.array(images)
    #labels = np.array(labels)


if __name__ == "__main__":
    main()
