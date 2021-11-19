from PIL import Image
from tensorflow import keras
import random
from tensorflow.keras import layers
import tensorflow as tf
import os
import numpy as np
import pydot
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'


ECG_model_shape = (4000,12,1)
DEM_ECFG_shape = (24,1)





class Ikem_beat(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size):
        self.batch_size = batch_size


    def __len__(self):
        return 10

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        x_1 = np.zeros((self.batch_size,) + ECG_model_shape, dtype="float32")
        x_2 = np.zeros((self.batch_size,) + DEM_ECFG_shape, dtype="float32")
        y = np.zeros((self.batch_size,) + (1,1), dtype="float32")
        return (x_2,x_1),y




def get_DEM_ECGF_model():
    inputs = keras.Input(shape=DEM_ECFG_shape, name="ECGF")
    x = layers.Dense(64)(inputs)
    x = layers.Dense(64)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(512)(x)
    return x,inputs


def get_ECG_model():
    inputs = keras.Input(shape=ECG_model_shape, name="ECG")
    x = layers.Conv2D(64, 3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    #Dot know pool size
    x = layers.MaxPooling2D(
    pool_size=(6, 6))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512,activation="relu")(x)
    return x,inputs


def complete_model():
    block_1_output,inputs_ECG = get_ECG_model()
    block_2_output,inputs_DEM = get_DEM_ECGF_model()

    block_1_fin = layers.add([block_1_output,block_2_output])
    x = layers.Dense(32)(block_1_fin)
    x = layers.Dense(32)(x)
    output = layers.Dense(1,activation="softmax")(x)
    model = keras.Model([inputs_DEM,inputs_ECG],output,name="DAVE_NEURON_NET")
    return model

model = complete_model()
model.summary()
model.compile(optimizer="adam",loss = tf.keras.losses.MeanAbsoluteError(
    reduction="auto", name="mean_absolute_error"
))
keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)

train_gen = Ikem_beat(1)

epochs = 10
model.fit(train_gen,epochs=epochs)