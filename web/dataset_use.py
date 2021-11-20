import pickle
from pathlib import Path

from tensorflow import keras
import tensorflow
import numpy as np

from main_train import get_training_set


class Predictor:
    def __init__(self,xml_model_path,pdf_model_path,vectorizer_path):
        self.xml_model = keras.models.load_model(xml_model_path)
        self.pdf_model = keras.models.load_model(pdf_model_path)
        self.vectorizer = self.load_vectoriter(vectorizer_path)

    def load_vectoriter(self,path):
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
            return vectorizer

    def predict(self,patient):
        if(patient.type == 2):
            data = np.ndarray.transpose(patient.data)

            to_predict = np.expand_dims(data,axis=0)
            x = self.xml_model.predict(to_predict)
            return x[0][0]
        if(patient.type == 1):

            string = patient.nlp
            #string = string.decode("utf-8")
            vec = self.vectorizer.transform([string]).toarray()
            to_predict = np.expand_dims(vec,axis=0)
            x = self.pdf_model.predict(to_predict)
            x = x[0][0][0]
            return x
        if(patient.type == 0):
            raise NotImplemented


if __name__ == "__main__":
    pred = Predictor("final.h5")
