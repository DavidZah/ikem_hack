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
from web.patient import Patient
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

word_vec_size = (1,47431)
batch_size = 32

class Ikem_npl(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,data,vec_vocab,model = 0):
        self.sentences = None
        self.batch_size = batch_size
        self.source = data
        #0 combined 1 wave form 2 pdf
        self.model = model
        self.vectorizer = vec_vocab


    def __len__(self):
       return len(self.source)//self.batch_size

    def __getitem__(self, idx):
        i = idx*batch_size
        batch_input_data = self.source[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + word_vec_size, dtype="float32")
        y = np.zeros((self.batch_size,) + (1,1), dtype="float32")
        for j, data in enumerate(batch_input_data):
                string = data.nlp
                string = string.decode("utf-8")
                vec = self.vectorizer.transform([string]).toarray()
                x[j] = vec
                y[j] = data.classification
        return x,y

def get_DEM_ECGF_model():
    inputs = keras.Input(shape=word_vec_size, name="WORD_VEC")
    x = layers.Dense(256)(inputs)
    x = layers.Dense(256)(x)
    x = layers.Dense(256)(x)
    x = layers.Dense(256)(x)
    x = layers.Dense(256)(x)
    x = layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model(inputs, x, name="DAVE_WORD_NET")
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
        if(i.type != 2):
            oper_lst.append(i)
    return oper_lst

if __name__ == "__main__":
    with open('../data/parrot.pkl', 'rb') as f:
        data = pickle.load(f)

    data = filetr_items(data)
    vectorizer = gen_vectored(data)

    with open(Path('../data/nlp_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    val_samples = 1
    train_data = data[:-val_samples]
    val_data = data[-val_samples:]


    npl = Ikem_npl(batch_size,train_data,vectorizer)
    npl_test = Ikem_npl(batch_size,val_data,vectorizer)

    model = get_DEM_ECGF_model()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000001), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy'])
    model.fit(npl,validation_data = npl_test,epochs=3)
    model.save(Path("../data/nlp_model.h5"))
    print("done")


