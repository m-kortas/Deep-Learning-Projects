import tensorflow as tf
#tf.enable_eager_execution()

A = tf.fill([10, 15], 1.0)
B = tf.range(1,16,1, dtype= tf.float32)
C = A * B

print(A)
print(B)
print(C)

print(C.shape)
print(C.dtype)

tf.cast(C, tf.uint8)
print(C < 7)
tf.where(C < 7, 7, C)
D = C.reduce_mean(axis=1)
E = C + D


sess = tf.Session()
ses_run = sess.run(C)
ses_run1 = sess.run(D)
ses_run2 = sess.run(E)
print(ses_run)
print(ses_run1)
print(ses_run2)
sess.close()

