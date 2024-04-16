import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


# 1)
unique_colors = np.unique(img_array, axis=0)
num_colors = unique_colors.shape[0]
print("Broj boja slike je: ", num_colors)


# 2)
K = 7

kmeans = KMeans(n_clusters=K)
fit_img = kmeans.fit_predict(img_array)
labels = kmeans.predict(img_array)


# 3)
for i in range(K):
    img_array_aprox[labels == i] = kmeans.cluster_centers_[i]
  
img_aprox = np.reshape(img_array_aprox, (w,h,d))

plt.figure()
plt.title("Aproksimirana slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

# 6)
intertia_values = []
K_values=range(1, 11)

for K in K_values:
    kmeans_int = KMeans(n_clusters=K)
    kmeans_int.fit(img_array)
    intertia_values.append(kmeans_int.inertia_)

plt.figure()
plt.plot(K_values, intertia_values)
plt.xlabel("K")
plt.ylabel("Intertija")
plt.title("Elbow metoda")
plt.show()

# 7)
for i in range(kmeans.n_clusters):
    binary_img = np.zeros((w,h,d))
    labels_reshaped = labels.reshape((w, h))
    binary_img[labels_reshaped == i] = 1
    plt.figure()
    plt.title("Grupa " + str(i+1))
    plt.imshow(binary_img)
    plt.show()