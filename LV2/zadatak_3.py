import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('road.jpg')

# a) 
img = img + 10
plt.figure()
plt.imshow(img)
plt.show()

# b)
img = plt.imread('road.jpg')
plt.figure()
plt.imshow(img[:, img.shape[1]//4:img.shape[1]//2])
plt.show()

# c) 
img = plt.imread('road.jpg')
img = np.rot90(img)
img = np.rot90(img)
img = np.rot90(img)
plt.figure()
plt.imshow(img)
plt.show()

# d) 
img = plt.imread('road.jpg')
img = img[:, ::-1]
plt.figure()
plt.imshow(img)
plt.show()



