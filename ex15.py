import numpy as np

np.random.seed(0)

A = np.random.rand(50)
B = np.random.rand(50)

A = np.around(A)
B = np.around(B)

A = np.uint8(A)
B = np.uint8(B)

print(A)
print(B)

TP = np.sum(np.bitwise_and(A == 1, B == 1))
TN = np.sum(np.bitwise_and(A == 0, B == 0))
FP = np.sum(np.bitwise_and(A == 0, B == 1))
FN = np.sum(np.bitwise_and(A == 1, B == 0))

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision*Recall) / (Precision+Recall)

print(Precision)
print(Recall)
print(F1)