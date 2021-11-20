import pickle

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
import random
from tensorflow.keras import layers
import tensorflow as tf
import os
import numpy as np
import pydot
import os
from web.patient import Patient
from pathlib import Path
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'


class Ikem_nlp_pdf(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,data,vectorizer,model = 0):
        self.batch_size = batch_size
        self.source = data
        #0 combined 1 wave form 2 pdf
        self.model = model
        self.vectorizer = vectorizer

    def __len__(self):
       return len(self.source) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size

        batch_input_data = self.source[i: i + self.batch_size]

        x_1 = np.zeros((self.batch_size,) + word_vec_size, dtype="float32")
        x_2 = np.zeros((self.batch_size,) + ECG_model_shape, dtype="float32")
        y = np.zeros((self.batch_size,) + (1,1), dtype="float32")
        for j,  data in enumerate(batch_input_data):
            string = data.nlp
            string = string.decode("utf-8")
            vec = self.vectorizer.transform([string]).toarray()
            x_1[j] = vec
            x_2[j] = np.expand_dims(data.data, 2)
            y[j] = data.classification
        return (x_1,x_2),y



def get_DEM_ECGF_model():
    inputs = keras.Input(shape=word_vec_size, name="ECGF")
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

def complete_model():
    block_1_output,inputs_ECG = get_ECG_model()
    block_2_output,inputs_DEM = get_DEM_ECGF_model()

    block_1_fin = layers.add([block_1_output,block_2_output])
    x = layers.Dense(32)(block_1_fin)
    x = layers.Dense(32)(x)
    output = layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model([inputs_DEM,inputs_ECG],output,name="DAVE_NEURON_NET")
    return model

def gen_vectored(data):
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    sentences = []
    for i in data:
        x = i.nlp
        sentences.append(x.decode("utf-8"))
    vectorizer.fit(sentences)
    return vectorizer

def filetr_items(data):
    oper_lst = []
    for i in data:
        if(i.type == 0):
            oper_lst.append(i)
    return oper_lst

word_vec_size = (1,47195)
ECG_model_shape = (12,5000,1)

batch_size = 16



if __name__ == "__main__":
    with open('../data/parrot.pkl', 'rb') as f:
        data = pickle.load(f)


    data = filetr_items(data)
    vectorizer = gen_vectored(data)

    with open(Path('../data/nlp_pdf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    val_samples = 1
    train_data = data[:-val_samples]
    val_data = data[-val_samples:]

    npl = Ikem_nlp_pdf(batch_size, train_data, vectorizer)
    npl_test = Ikem_nlp_pdf(batch_size, val_data, vectorizer)

    model = complete_model()
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])
    model.fit(npl, validation_data=npl_test, epochs=2)
    model.save_weights('../web/classifier/pdf_nlp.h5')
    print("done")