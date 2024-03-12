import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
# a)
print("Mjerenje je izvrseno na " + str((len(data))) + " osoba")

# b)
height = data[:, 1]
weight = data[:, 2]
plt.scatter(height, weight)
plt.xlabel("Visina")
plt.ylabel("Tezina")
plt.title("Odnos visine i tezine ispitanika")
plt.show()

# c)
height = data[::50,1]
weight = data[::50,2]
plt.scatter(height, weight)
plt.xlabel("Visina")
plt.ylabel("Tezina")
plt.title("Odnos visine i tezine svakog pedesetog ispitanika")
plt.show()

# d)
print("Najmanja visina je: " + str(np.min(data[:,1])))
print("Najveca visina je: " + str(np.max(data[:,1])))
print("Prosjecna visina je: " + str(np.mean(data[:,1])))

# e)
ind = (data[:,0] == 1)
print("Najmanja visina muskaraca: " + str(np.min(data[ind,1])))
print("Najveca visina muskaraca: " + str(np.max(data[ind,1])))
print("Prosjecna visina muskaraca: " + str(np.mean(data[ind,1])))

ind = (data[:,0] == 0)  
print("Najmanja visina zena: " + str(np.min(data[ind,1])))
print("Najveca visina zena: " + str(np.max(data[ind,1])))
print("Prosjecna visina zena: " + str(np.mean(data[ind,1])))
