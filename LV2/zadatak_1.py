import numpy as np
import matplotlib.pyplot as plt

x_coords = np.array([1, 3, 3, 2, 1])
y_coords = np.array([1, 1, 2, 2, 1])

plt.plot(x_coords, y_coords, linewidth=2, color='r')
plt.axis([0, 4, 0, 4])
plt.title('Slika 2.3')
plt.xlabel('x')
plt.ylabel('y')
plt.show()