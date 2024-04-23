from tensorflow import keras
from keras import layers

model = keras.Sequential()
model.add(layers.Input(shape=(2,)))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()