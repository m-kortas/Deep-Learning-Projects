import tensorflow as tf

# wejście
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
# to co powinno wyjść
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

# sieć neuronowa składająca się z 1 warstwy i 1 neuronu
linear_model = tf.layers.Dense(units=1)

# wyjście sieci neuronowej
y_pred = linear_model(x)
# funkcja kosztu
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
# optymalizator
optimizer = tf.train.GradientDescentOptimizer(0.01)
# definicja treningu - minimalizacja lossu
train = optimizer.minimize(loss)
# inicjalizacja wag
init = tf.global_variables_initializer()
# stworzenie sesji
sess = tf.Session()
# uruchomienie inicjalizacji
sess.run(init)
# 1000 epok
for i in range(1000):
    # wykonaj trening i policz loss dla kolejnej epoki
    _, loss_value = sess.run((train, loss))
    # wypisz loss
    print(loss_value)

# wypisz predykcję sieci
print(sess.run(y_pred))
# zakończ sesję
sess.close()
