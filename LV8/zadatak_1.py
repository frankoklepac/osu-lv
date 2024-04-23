import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras import utils as image
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(5, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
plt.show()


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

x_train_vector = x_train_s.reshape(60000, 784)
x_test_vector = x_test_s.reshape(10000, 784)

# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape = (784,)))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# TODO: provedi ucenje mreze
model.fit(x_train_vector, y_train_s, batch_size=32, epochs=20, validation_split=0.1)

# TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate(x_test_vector, y_test_s, verbose=0)
conf_mat = confusion_matrix(y_test, np.argmax(model.predict(x_test_vector), axis=-1))
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Confusion matrix:")
print(conf_mat)

# TODO: spremi model
model.save("LV8_model.keras")

# Ovdje se nalazi rjesenje zadatka 2 zato sto kada prebacim na Python 3.8 VS Code ne može naći keras.models import

x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

num_classes = 10
y_test_s = keras.utils.to_categorical(y_test, num_classes)
x_test_vector = x_test_s.reshape(-1, 784)

predictions = model.predict(x_test_vector)

for i in range(300):
    if y_test[i] != predictions[i].argmax():
        plt.imshow(x_test[i], cmap="gray")
        plt.title(f"Stvarna oznaka: {y_test[i]}, Predvidjena oznaka: {predictions[i].argmax()}")
        plt.show()

# Ovdje se nalazi rjesenje zadatak 3, razlog isti isti kao i kod zadatka 2

img = image.load_img("LV8\example-number-7.jpg", target_size = (28, 28), color_mode = "grayscale")
img_array = image.img_to_array(img)

img_array = img_array.astype("float32") / 255
img_array_s = np.expand_dims(img_array, -1)

img_array_vector = img_array_s.reshape(-1, 784)

prediction = model.predict(img_array_vector)


plt.imshow(img_array, cmap="gray")
plt.title(f"Predvidjena oznaka: {prediction.argmax()}")
plt.show()