import cv2
import keras
import numpy as np
from keras.applications import ResNet50, VGG16
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, np, BatchNormalization, Activation, regularizers, Dropout
from matplotlib import pyplot as plt
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency, visualize_cam, overlay


def main():
    model = VGG16()
   # model.summary()

    img = cv2.imread("/Users/ja/Deep-Learning-Projects/dog.jpg")
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    layer_idx = utils.find_layer_idx(model, 'predictions')
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    heatmap = visualize_cam(model, layer_idx, filter_indices = 162, seed_input=img)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    attention_img = overlay(heatmap, img)

    cv2.imshow("attention", attention_img)
    cv2.waitKey()

    #x = model.predict(np.array([img]))[0]
    #print(np.argsort(x)[-5:])


if __name__ == '__main__':
    main()



