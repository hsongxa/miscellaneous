# This is the second example program from the 'Introduction
# to "Learning Deep Learning"' training course at:
# https://www.nvidia.com/en-us/on-demand/session/gtcspring23-dlit52044/?playlistId=playList-5906792e-0f3d-436b-bc30-3abf911a95a6
# by Magnus Ekman.

import numpy as np

def neuron_w(input_count):
  weights = np.zeros(input_count)
  for i in range(1, input_count):
    weights[i] = np.random.uniform(-1.0, 1.0)
  return weights

def show_learning():
  print("Current weights:")
  for i, w in enumerate(n_w):
    print("neuron ", i, ": w0 =", "%5.2f" % w[0],
          ", w1 =", "%5.2f" % w[1], ", w2 =", "%5.2f" % w[2])
  print("------------------------")

def forward_pass(x):
  global n_y
  n_y[0] = np.tanh(np.dot(n_w[0], x)) # neuron 0
  n_y[1] = np.tanh(np.dot(n_w[1], x)) # neuron 1
  n2_inputs = np.array([1.0, n_y[0], n_y[1]]) # 1.0 is bias
  z2 = np.dot(n_w[2], n2_inputs)
  n_y[2] = 1.0 / (1.0 + np.exp(-z2)) # sigmoid function

def backward_pass(y_truth):
  global n_error
  error_prime = n_y[2] - y_truth
  derivative = n_y[2] * (1.0 - n_y[2]) # logistic derivative
  n_error[2] = error_prime * derivative
  derivative = 1.0 - n_y[0]**2 # tanh derivative
  n_error[0] = n_w[2][1] * n_error[2] * derivative
  derivative = 1.0 - n_y[1]**2 # tanh derivative
  n_error[1] = n_w[2][2] * n_error[2] * derivative

def adjust_weights(x):
  global n_w
  n_w[0] -= (x * learning_rate * n_error[0])
  n_w[1] -= (x * learning_rate * n_error[1])
  n2_inputs = np.array([1.0, n_y[0], n_y[1]])
  n_w[2] -= (n2_inputs * learning_rate * n_error[2])

np.random.seed(3)
index_list = [0, 1, 2, 3]
learning_rate = 0.1

x_train = [np.array([1.0, -1.0, -1.0]),
           np.array([1.0, -1.0, 1.0]),
           np.array([1.0, 1.0, -1.0]),
           np.array([1.0, 1.0, 1.0])]
y_train = [0.0, 1.0, 1.0, 0.0]

n_w = [neuron_w(3), neuron_w(3), neuron_w(3)]
n_y = [0, 0, 0]
n_error = [0, 0, 0]
 
show_learning()

for i in range(1000):
  np.random.shuffle(index_list)
  for j in index_list:
    forward_pass(x_train[j])
    backward_pass(y_train[j])
    adjust_weights(x_train[j])
show_learning()

for i in range(len(x_train)):
  forward_pass(x_train[i])
  print("x0 =", "%4.1f" % x_train[i][0], ", x1 =", "%4.1f" % x_train[i][1],
        ", x2 =", "%4.1f" % x_train[i][2], ", y =", "%4.1f" % n_y[2])
