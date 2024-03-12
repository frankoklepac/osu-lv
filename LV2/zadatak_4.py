import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50))
white = np.ones((50, 50))

top_row = np.hstack((black, white))
bottom_row = np.hstack((white, black))

image = np.vstack((top_row, bottom_row))
plt.imshow(image, cmap='gray')
plt.show()