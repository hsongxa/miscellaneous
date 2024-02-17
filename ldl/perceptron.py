# This is the first example program from the 'Introduction
# to "Learning Deep Learning"' training course at:
# https://www.nvidia.com/en-us/on-demand/session/gtcspring23-dlit52044/?playlistId=playList-5906792e-0f3d-436b-bc30-3abf911a95a6
# by Magnus Ekman.

import random

def show_learning(w):
  print("w0 =", "%5.2f" % w[0],
        "w1 =", "%5.2f" % w[1],
        "w2 =", "%5.2f" % w[2])

def compute_output(w, x):
  z = 0.0
  for i in range(len(w)):
    z += x[i] * w[i]
  if (z < 0):
    return -1
  else:
    return 1

random.seed(7)
index_list = [0, 1, 2, 3]
learning_rate = 0.1

x_train = [(1.0, -1.0, -1.0),
           (1.0, -1.0, 1.0),
           (1.0, 1.0, -1.0),
           (1.0, 1.0, 1.0)]
y_train = [1.0, 1.0, 1.0, -1.0]
w = [0.2, -0.6, 0.25]
show_learning(w)

all_correct = False
while (not all_correct):
  all_correct = True
  random.shuffle(index_list)
  for i in index_list:
    x = x_train[i]
    y = y_train[i]
    p_out = compute_output(w, x)
    if (y != p_out):
      for j in range(0, len(w)):
        w[j] += (y * learning_rate * x[j])
      all_correct = False
      show_learning(w)

print(compute_output(w, (1.0, -1.0, -1.0)))
