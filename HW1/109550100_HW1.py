# ----------------Linear regression------------------------

import numpy as np 
import matplotlib.pyplot as plt

def mean_square_error(y_pred, y_train):
    return (1 / y_pred.size) * sum((y_pred - y_train) * (y_pred - y_train))

x_train, x_test, y_train, y_test = np.load('regression_data.npy', allow_pickle=True)
# plt.plot(x_train, y_train, '.')
x_train = np.reshape(x_train, x_train.size)
x_test = np.reshape(x_test, x_test.size)

# Building the model
b1 = np.random.normal(0,1)
b0 = np.random.normal(0,1)

L = 0.01  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(x_train)) # Number of elements in X

loss_train = [] # loss of training data and prediction
loss_test = [] # loss of validation data and prediction

# Performing Gradient Descent 
for i in range(epochs): 
    y_pred = b1 * x_train + b0  # The current predicted value of Y
    loss_train.append(mean_square_error(y_pred, y_train))

    # calculate gradient
    D_b1 = (-2 / n) * sum(x_train * (y_train - y_pred))
    D_b0 = (-2 / n) * sum(x_train - y_pred) 

    b1 = b1 - L * D_b1  # Update b1
    b0 = b0 - L * D_b0  # Update b0

    y_pred = b1 * x_test + b0  # The current predicted value of Y
    loss_test.append(mean_square_error(y_pred, y_test))

# Making predictions
# y_pred = b1 * x_train + b0

# plt.plot(range(epochs), loss_train, color='blue')
# plt.plot(range(epochs), loss_test, color='red')

y_pred = b1 * x_test + b0  # The current predicted value of Y
print("weight:", b1, "; intercept:", b0)
print("Mean Square Error:", mean_square_error(y_pred, y_test))

plt.plot(x_test, y_test, '.')
plt.plot(x_test, y_pred, '.')

plt.show()

# ----------------logistic regression------------------------

def sigmoid(z): # sigmoid function
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z): # derivative of sigmoid function
    return sigmoid(z) * (1 - sigmoid(z))

def cross_entropy_loss(_y_pred, _y_train):
    return -(1 / _y_pred.size) * np.sum(_y_train * np.log(_y_pred + 1e-100) + (1 - _y_train) * np.log(1 - _y_pred + 1e-100))

def gradient_descent(y_pred, y_train, x_train):
    pe_pb0 = -np.sum((y_train - y_pred) / (y_pred * (1 - y_pred)) * y_pred * (1 - y_pred) * 1)
    pe_pb1 = -np.sum((y_train - y_pred) / (y_pred * (1 - y_pred)) * y_pred * (1 - y_pred) * x_train)
    return (pe_pb1, pe_pb0)

x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)

x_train = np.reshape(x_train, x_train.size)
x_test = np.reshape(x_test, x_test.size)

L = 0.001  # The learning Rate
epochs = 5000  # The number of iterations to perform gradient descent

# Building the model
b1 = np.random.normal(0,1)
b0 = np.random.normal(0,1)

loss_train = [] # loss of training data and prediction
loss_test = [] # loss of validation data and prediction

for i in range(epochs): 
    z = b1 * x_train + b0  # The current predicted value of Y
    s_y_pred = sigmoid(z)
    s_y_train = y_train
    loss_train.append(cross_entropy_loss(s_y_pred, s_y_train))

    # calculate gradient
    (gradient_b1, gradient_b0) = gradient_descent(s_y_pred, s_y_train, x_train)

    # update w
    b1 = b1 - L * gradient_b1
    b0 = b0 - L * gradient_b0

    z = b1 * x_test + b0  # The current predicted value of Y
    s_y_pred = sigmoid(z)
    s_y_test = y_test
    loss_test.append(cross_entropy_loss(s_y_pred, s_y_test))

z = b1 * x_test + b0  # The current predicted value of Y
s_y_pred = sigmoid(z)
s_y_test = y_test
print("weight:", b1, "; intercept:", b0)
print("Cross Entropy Loss:", cross_entropy_loss(s_y_pred, s_y_test))

plt.plot(range(epochs), loss_train, color='blue')
plt.plot(range(epochs), loss_test, color='red')

# plt.scatter(x_train, y_train)
plt.show()