# gradient decent is intelligence

import matplotlib.pyplot as plt
import numpy as np


def y_function(x):
    return x**2


def y_derivative(x):
    return 2 * x


current_position = (88, y_function(88))

learning_rate = 0.001

for _ in range(1000):
    new_x = current_position[0] - learning_rate * y_derivative(current_position[0])
    new_y = y_function(new_x)
    new_position = (new_x, new_y)

x = np.arange(-100, 100, 0.1)
y = y_function(x)


plt.plot(x, y)
plt.scatter(current_position[0], current_position[1], color="red")
plt.pause(0.001)
plt.clf()
