import os
import cv2
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, np, BatchNormalization, Activation, regularizers
from keras.utils import np_utils


TRAIN_LABELS_FILE = "faces/faces_is/train/labels.txt"
VAL_LABELS_FILE = "faces/faces_is/val/labels.txt"

def read_labels(filename, number):
    with open(filename) as f:
        lines = f.read().splitlines()[:number]
    return lines


def create_dataset(lines, dirname):
    images = []
    labels = []

    for line in lines:
        path, label = line.split()
        path = os.path.join(dirname, path)
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        images.append(img)

        label = int(label)
        label = np_utils.to_categorical(label, 52)
        labels.append(label)

    return np.array(images), np.array(labels)


def show_dataset(images, labels):
    for img, l in zip(images, labels):
        print(l)
        cv2.imshow("image", img)
        cv2.waitKey()


def create_model():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(16, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(52, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='sgd')
   # model.load_weights("weights.hdf5")
    return model



def get_callbacks():
    callbacks = []

    mc = ModelCheckpoint("weights.hdf5", monitor="val_loss", save_best_only=True, verbose=1)
    es = EarlyStopping(monitor="val_loss", patience=2, verbose=1)
    ts = TensorBoard()
    callbacks.append(mc)
    callbacks.append(ts)
    callbacks.append(es)

    return callbacks


def main():
    train_lines = read_labels(TRAIN_LABELS_FILE, 100)
    train_images, train_labels = create_dataset(train_lines, os.path.dirname(TRAIN_LABELS_FILE))

    val_lines = read_labels(VAL_LABELS_FILE, 50)
    val_images, val_labels = create_dataset(val_lines, os.path.dirname(VAL_LABELS_FILE))

    callbacks = get_callbacks()

    model = create_model()
    model.summary()
    model.fit(train_images, train_labels, batch_size=10, epochs=100, validation_data=(val_images, val_labels), callbacks=callbacks)


if __name__ == '__main__':
    main()