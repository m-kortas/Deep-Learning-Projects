import random

from keras import Sequential
from keras.layers import Dense
from keras.utils import get_file
import numpy as np


def get_chars_words(word_len):
    path = get_file('pantadeusz.txt', origin='https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt')
    with open(path, encoding='utf-8') as f:
        text = f.read().lower()
    words = text.split()
    words = set([word for word in words if len(word) == word_len])
    chars = set([letter for word in words for letter in word])
    return sorted(chars), list(words)


def get_data(words, map_):
    x = np.zeros((len(words), len(words[0]), len(map_)), dtype=np.float32)
    for i, word in enumerate(words):
        for j, ch in enumerate(word):
            x[i, j, map_[ch]] = 1.0

    return x, x


def create_model(input_size):
    model = Sequential()
    model.add(Dense(10, input_shape=input_size))
    model.add(Dense(3))
    model.add(Dense(10))
    model.add(Dense(input_size[1], activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.summary()

    return model


def check_model(model, words, char_to_int, int_to_char):
    idx = random.randint(0, len(words))
    chosen_word = words[idx]

    x2, _ = get_data([chosen_word], char_to_int)
    result = model.predict(x2)[0]
    result = np.round(result)

    predicted_word = "".join([int_to_char[np.argmax(i)] for i in result])

    print(chosen_word)
    print(predicted_word)


def main():
    word_len = 5
    chars, words = get_chars_words(word_len)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    x, y = get_data(words, char_to_int)
    model = create_model((word_len, len(char_to_int)))

    model.fit(x, y, batch_size=10, epochs=10)

    check_model(model, words, char_to_int, int_to_char)


if __name__ == '__main__':
    main()
