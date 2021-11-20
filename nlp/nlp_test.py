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

batch_size = 32
seed = 42
max_features = 10000
sequence_length = 250

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.transform(sentences).toarray()
x = vectorizer.transform(sentences).toarray()
print(vectorizer.vocabulary_)

class Ikem_npl(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,data,model = 0):
        self.batch_size = batch_size
        self.source = data
        #0 combined 1 wave form 2 pdf
        self.model = model

        self.vectorizer = CountVectorizer(min_df=0, lowercase=False)

    def __len__(self):
       return 1

    def __getitem__(self, idx):




        x_2 = np.zeros((self.batch_size,) + (250,), dtype="float32")
        y = np.zeros((self.batch_size,) + (1,1), dtype="float32")
        x_2 = ["str"]
        return x_2,y

with open('../data/parrot.pkl', 'rb') as f:
    data = pickle.load(f)

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(tf.keras.layers.Embedding(max_features + 1, sequence_length))
model.add(layers.Dense(128))
model.add(layers.Dense(128))
model.add(layers.Dense(1))
model.compile(optimizer="adam",loss =tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy'])
model.summary()
train = Ikem_npl(1,1)
input_data = [["foo qux bar"], ["qux baz"]]
model.predict(input_data)