from tensorflow import keras
import tensorflow
import numpy as np

from layers_test import get_training_set


class Predictor:
    def __init__(self,model_path):
        self.model = keras.models.load_model(model_path)
    def predict(self,patient):
        data = np.ndarray.transpose(patient.data)

        to_predict = np.expand_dims(data,axis=0)
        x = self.model.predict(to_predict)
        return x[0][0]

if __name__ == "__main__":
    pred = Predictor("final.h5")
