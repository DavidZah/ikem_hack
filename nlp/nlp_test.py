import pickle
import re
import string

from keras import losses
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
from sklearn.feature_extraction.text import CountVectorizer

sentences = ["tohle je náhodná věta","tohle není náhodná věta"]
target = [1,0]

word_vec_size = (47431,1)
batch_size = 32

class Ikem_npl(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,data,model = 0):
        self.sentences = None
        self.batch_size = batch_size
        self.source = data
        #0 combined 1 wave form 2 pdf
        self.model = model

        self.__filetr_items__()
        self.__gen_vectored__()
        print(len(self.vectorizer.vocabulary_))

    def __len__(self):
       return 1

    def __getitem__(self, idx):

        x_2 = np.zeros((self.batch_size,) + word_vec_size, dtype="float32")
        y = np.zeros((self.batch_size,) + (1,1), dtype="float32")

        return x_2,y


    def __filetr_items__(self):
        oper_lst = []
        for i in self.source:
            if(i.type != 2):
                oper_lst.append(i)
        self.source = oper_lst

    def __gen_vectored__(self):
        self.vectorizer = CountVectorizer(min_df=0, lowercase=False)
        self.sentences = []
        for i in self.source:
            x = i.nlp
            self.sentences.append(x.decode("utf-8"))
        self.vectorizer.fit(self.sentences)

with open('../data/parrot.pkl', 'rb') as f:
    data = pickle.load(f)

npl = Ikem_npl(1,data)

def get_DEM_ECGF_model():
    inputs = keras.Input(shape=word_vec_size, name="WORD_VEC")
    x = layers.Dense(64)(inputs)
    x = layers.Dense(64)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model(inputs, x, name="DAVE_WORD_NET")
    return model



model = get_DEM_ECGF_model()

model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])
model.fit(npl,epochs=10)
print("done")


