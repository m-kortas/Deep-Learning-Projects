import io
import random
import sys

import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils.data_utils import get_file

# pobranie pliku z Panem Tadeuszem
path = get_file('pantadeusz.txt', origin='https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()


# lista wszystkich występujących znaków
chars = sorted(set(text))


# mapowanie ze znaku na liczbę i odwrotne
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

sen_len = 40
step = 3

sentences = []
next_chars = []

# dane treningowe x = zdanie -> y = kolejna literka
for i in range(0, len(text) - sen_len, step):
    sentences.append(text[i: i + sen_len])
    next_chars.append(text[i + sen_len])
print('sequences:', len(sentences))

# przygotowanie danych do sieci - tablice numpy one hot
x = np.zeros((len(sentences), sen_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# model z jedną warstwą LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(sen_len, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# pewien rodzaj magii, pozwala zwrócić nam kolejną literkę, jedną z najbardziej prawdopodobnych
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    # "odwrócenie softmax"
    preds = np.log(preds) / temperature

    # "softmax"
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # wylosowanie wartości z rozkładu
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# funkcja będzie wywoływana na koniec każdej epoki
def on_epoch_end(epoch, logs):
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    # z ktorego miejsca wziąć początek do generowania
    start_index = random.randint(0, len(text) - sen_len - 1)

    # wygenerowany do tej pory tekst
    generated = ''
    # początek do generowania
    sentence = text[start_index: start_index + sen_len]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        # kodowanie zdania do predykcji
        x_pred = np.zeros((1, sen_len, len(chars)))
        for t, ch in enumerate(sentence):
            x_pred[0, t, char_to_index[ch]] = 1.

        # predykcja
        preds = model.predict(x_pred, verbose=0)[0]
        # wybranie kolejnej literki - brak tej funkcji powoduje generowanie cigle tego samego tekstu
        next_index = sample(preds, 0.2)
        next_char = index_to_char[next_index]

        # dodanie kolejnej literki do wygenerowanego tekstu
        generated += next_char
        # "przesunięcie" zdania o 1 w prawo
        sentence = sentence[1:] + next_char

        # wymuszenie wypisania na ekran
        sys.stdout.write(next_char)
        sys.stdout.flush()


# dodatkowy callback, żeby wypisywać tekst po każdej epoce
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=100, callbacks=[print_callback])
