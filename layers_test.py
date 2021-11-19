from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras
import random
from tensorflow.keras import layers
import tensorflow as tf
import os
import numpy as np
import pydot
import os
from patient import Patient

os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'


ECG_model_shape = (12,5000,1)
DEM_ECFG_shape = (24,1)




def get_training_set(json_file, npy_path):
    return_list = []
    npy_filelist = os.listdir(npy_path)
    count = 0
    for filename in npy_filelist:
        new_patient = Patient(npy_path+filename, json_file)
        if new_patient.classification == None:
            continue
        count += 1
        return_list.append(new_patient)
    print(count)
    return return_list

class Ikem_beat(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,data,model = 0):
        self.batch_size = batch_size
        self.source = data
        #0 combined 1 wave form 2 pdf
        self.model = model


    def __len__(self):
       return len(data) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size

        batch_input_data = self.source[i: i + self.batch_size]

        if(self.model == 0):
            raise NotImplementedError
        if(self.model == 1):
            x_2 = np.zeros((self.batch_size,) + ECG_model_shape, dtype="float32")
            y = np.zeros((self.batch_size,) + (1,1), dtype="float32")
            for j,  data in enumerate(batch_input_data):
                x_2[j] = np.expand_dims(data.data[2],2)
                y[j] = data.classification
            return x_2,y
        if (self.model == 2):
            raise NotImplementedError

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
    x = layers.Conv2D(64, 1, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    #Dot know pool size
    x = layers.MaxPooling2D(
    pool_size=(3, 3))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512,activation="relu")(x)
    return x,inputs

def get_wave_form_model():
    block_1_output,inputs_ECG = get_ECG_model()

    x = layers.Dense(32)(block_1_output)
    x = layers.Dense(32)(x)
    output = layers.Dense(1,activation="softmax")(x)
    model = keras.Model(inputs_ECG, output, name="DAVE_NEURON_NET")
    return model

def complete_model():
    block_1_output,inputs_ECG = get_ECG_model()
    block_2_output,inputs_DEM = get_DEM_ECGF_model()

    block_1_fin = layers.add([block_1_output,block_2_output])
    x = layers.Dense(32)(block_1_fin)
    x = layers.Dense(32)(x)
    output = layers.Dense(1,activation="softmax")(x)
    model = keras.Model([inputs_DEM,inputs_ECG],output,name="DAVE_NEURON_NET")
    return model


data = get_training_set("data\\class2.json","data\\npy\\")

#model = complete_model()
model = get_wave_form_model()
model.summary()
model.compile(optimizer="adam",loss =tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)

random.shuffle(data)

val_samples = 200
train_gen = Ikem_beat(1,data[:-val_samples],1)
valid_gen = Ikem_beat(1,data[-val_samples:],1)

callbacks = [
    keras.callbacks.ModelCheckpoint("callback_save.h5", save_best_only=True)
]

epochs = 7
history = model.fit(train_gen,validation_data =  valid_gen,epochs=epochs,callbacks=callbacks)

model.save("final.h5")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()