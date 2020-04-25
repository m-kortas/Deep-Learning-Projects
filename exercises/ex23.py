import tensorflow as tf

# 1
tf.enable_eager_execution()

# 2
a = tf.fill((10, 15), 1.0)
# print(a)

# 3
b = tf.range(1, 16, dtype=tf.float32)
# print(b)

# 4
c = a * b
# print(c)

# 5
print(c.shape, c.dtype)

# 6
c = tf.cast(c, tf.uint8)
# print(c.dtype)

# 7
idx = tf.where(c < 7)
print(idx)

# 8
c = tf.where(c < 7, 7 * tf.ones_like(c), c)
# print(c)

# 9
diag = tf.eye(10, 15, dtype=tf.uint8)
c += diag
# print(c)

# 10
avg_row = tf.reduce_mean(c, axis=0)
# print(avg_row)

# 11
avg_row = tf.reshape(avg_row, (1, -1))
c = tf.concat((c, avg_row), axis=0)
print(c)
