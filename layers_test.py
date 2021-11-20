
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
from pathlib import Path
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

batch_size = 1

ECG_model_shape = (12,5000,1)
DEM_ECFG_shape = (24,1)


def get_training_and_validation_set(json_file, npy_path, tp=15):
    validation_set = []
    training_set =[]
    orig_training_set = get_training_set(json_file, npy_path)
    sick_counter = 0
    healthy_counter = 0
    for patient in orig_training_set:
        if patient.classification == 1 and sick_counter != tp:
            validation_set.append(patient)
            sick_counter += 1
        elif patient.classification == 0 and healthy_counter != tp:
            validation_set.append(patient)
            healthy_counter += 1
        else:
            training_set.append(patient)
    return training_set, validation_set

def get_training_and_validation_set_giv_size(json_file, npy_path, tp=10, pp = 150):
    validation_set = []
    training_set =[]
    orig_training_set = get_training_set(json_file, npy_path)
    sick_counter = 0
    healthy_counter = 0
    patient_counter = 0

    for patient in orig_training_set:
        if patient.classification == 1 and sick_counter != tp:
            validation_set.append(patient)
            sick_counter += 1
        elif patient.classification == 0 and healthy_counter != tp:
            validation_set.append(patient)
            healthy_counter += 1
        elif patient.classification == 1 and patient_counter != pp*2:
            training_set.append(patient)
            patient_counter +=1
        elif patient.classification == 0 and patient_counter != pp * 2:
            training_set.append(patient)
            patient_counter +=1
    return training_set, validation_set

def get_training_set(json_file, npy_path):
    npy_path += os.path.sep
    return_list = []
    npy_filelist = os.listdir(npy_path)
    count = 0
    for filename in npy_filelist:
        new_patient = Patient(npy_path+filename, json_file)
        if new_patient.classification == None:
            continue
        count += 1
        return_list.append(new_patient)
    return return_list

class Ikem_beat(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,data,model = 0):
        self.batch_size = batch_size
        self.source = data
        #0 combined 1 wave form 2 pdf
        self.model = model


    def __len__(self):
       return len(self.source) // self.batch_size

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
    x = layers.Conv2D(64, 2, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    tf.keras.layers.Dropout(0.2)
    #Dot know pool size
    x = layers.MaxPooling2D(
    pool_size=(3, 3))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512,activation="relu")(x)
    return x,inputs

def get_wave_form_model():
    block_1_output,inputs_ECG = get_ECG_model()

    x = layers.Dense(64)(block_1_output)
    x = layers.Dense(64)(x)
    output = layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model(inputs_ECG, output, name="DAVE_NEURON_NET")
    return model

def mega_cool_neural_net():
    inputs = keras.Input(shape=ECG_model_shape, name="COOLmodel")


    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(inputs)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)

    return keras.models.Model(inputs=inputs, outputs=output_layer)



def complete_model():
    block_1_output,inputs_ECG = get_ECG_model()
    block_2_output,inputs_DEM = get_DEM_ECGF_model()

    block_1_fin = layers.add([block_1_output,block_2_output])
    x = layers.Dense(32)(block_1_fin)
    x = layers.Dense(32)(x)
    output = layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model([inputs_DEM,inputs_ECG],output,name="DAVE_NEURON_NET")
    return model


if __name__ == "__main__":

    data = get_training_set(str(Path("data/class2_2.json")),str(Path("data/npy/")))

    train_data , val_data = get_training_and_validation_set_giv_size(str(Path("data/class2.json")),str(Path("data/npy/")))

    random.shuffle(train_data)
    random.shuffle(val_data)

    #model = complete_model()
    model = get_wave_form_model()
    #model = mega_cool_neural_net()
    model.summary()
    model.compile(optimizer="adam",loss =tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy'])
    #keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)

    random.seed()
    random.shuffle(data)


    val_samples = 250

    #train_data = data[:-val_samples]
    #val_data = data[-val_samples:]
    print(f"Val data is {len(val_data)}")
    print(f"Train data is {len(train_data)}")
    train_gen = Ikem_beat(batch_size,train_data,1)
    valid_gen = Ikem_beat(batch_size,val_data,1)

    callbacks = [
        keras.callbacks.ModelCheckpoint("callback_save.h5", save_best_only=True)
    ]

    epochs = 15
    history = model.fit(train_gen,validation_data =  valid_gen,epochs=epochs)

    model.save("final.h5")

