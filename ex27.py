from keras import Sequential
from keras.datasets import imdb
from keras.layers import Embedding, Dense, SimpleRNN
from keras_preprocessing import sequence


def get_model(max_features):
    model = Sequential()
    model.add(Embedding(max_features, output_dim=256))
    model.add(SimpleRNN(128, activation="tanh"))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    max_words = 10000
    max_len = 100

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    model = get_model(max_words)
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
