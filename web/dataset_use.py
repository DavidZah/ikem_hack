from tensorflow import keras
import tensorflow
import numpy as np

from main_train import get_training_set


class Predictor:
    def __init__(self,xml_model_path,pdf_model_path):
        self.xml_model = keras.models.load_model(xml_model_path)
        self.pdf_model = keras.models.load_model(pdf_model_path)
        self.vectorizer =

    def load_vectoriter(self,path):

    def predict(self,patient):
        data = np.ndarray.transpose(patient.data)

        to_predict = np.expand_dims(data,axis=0)
        x = self.model.predict(to_predict)
        return x[0][0]

if __name__ == "__main__":
    pred = Predictor("final.h5")
