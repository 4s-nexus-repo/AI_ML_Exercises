#step1:
age = 30
salary = 70

w1 = 0.1
w2 = 0.2
b = 1

z = (w1 * age) + (w2 * salary) + b

def relu(z):
    return max(0, z)

a = relu(z)

print("Raw neuron value z:", z)
print("After ReLU activation:", a)


#step2:
import numpy as np

X = np.array([30, 70])  # age, salary

# weights for 3 neurons (3x2)
W = np.array([
    [0.1, 0.2],
    [0.05, 0.05],
    [-0.2, 0.05]
])

b = np.array([1, 0, 1])

Z = np.dot(W, X) + b

def relu(z):
    return np.maximum(0, z)

A = relu(Z)

print("Z:", Z)
print("A (after ReLU):", A)


import numpy as np

A = np.array([4, 4, 1])  # hidden output

W_out = np.array([0.2, 0.3, 0.1])
b_out = 0

z = np.dot(W_out, A) + b_out

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

output = sigmoid(z)

print("z:", z)
print("probability:", output)

if output > 0.5:
    print("BUY")
else:
    print("NOT BUY")