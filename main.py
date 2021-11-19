from PIL import Image
from tensorflow import keras
import random
from tensorflow.keras import layers
import tensorflow as tf
import os
import numpy as np


ECG_model_shape = (4000,12,1)
batch_size = 1
shape = (1,1)


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size):
        self.batch_size = batch_size


    def __len__(self):
        return 10

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        x = np.zeros((self.batch_size,) + ECG_model_shape, dtype="float32")
        y = np.zeros((self.batch_size,) + (1,), dtype="float32")
        return x,y






def get_ECG_model():
    inputs = keras.Input(ECG_model_shape)
    x = layers.Conv2D(64,activation="relu")(inputs)

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()



# Build model
model = get_model()
model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.accuray )
model.summary()


val_samples = 5

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    1, img_size,
)



# Train the model, doing validation at the end of each epoch.
epochs = 10
model.fit(train_gen,epochs=epochs)

model.save("model_bleh")

