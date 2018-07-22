import os

import cv2
from keras.utils import np_utils

LABELS_FILE = "faces/faces_is/train/labels.txt"


def read_labels(filename, number):
    with open(filename) as f:
        lines = f.read().splitlines()[:number]
    return lines


def create_dataset(lines):
    images = []
    labels = []

    dirname = os.path.dirname(LABELS_FILE)
    for line in lines:
        path, label = line.split()
        path = os.path.join(dirname, path)
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        images.append(img)

        label = int(label)
        label = np_utils.to_categorical(label, 52)
        labels.append(label)

    return images, labels


def show_dataset(images, labels):
    for img, l in zip(images, labels):
        print(l)
        cv2.imshow("image", img)
        cv2.waitKey()


def main():
    lines = read_labels(LABELS_FILE, 100)
    images, labels = create_dataset(lines)

    show_dataset(images, labels)


if __name__ == '__main__':
    main()