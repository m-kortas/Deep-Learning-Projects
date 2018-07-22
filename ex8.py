import os

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, np
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


def create_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(52, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


def main():
    lines = read_labels(LABELS_FILE, 100)
    images, labels = create_dataset(lines)

    images = np.array(images)
    labels = np.array(labels)
    model = create_model()
    model.fit(images, labels, batch_size=1, epochs=10)


if __name__ == '__main__':
    main()