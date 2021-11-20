from tensorflow import keras
import tensorflow
import numpy as np

from layers_test import get_training_set


class Predictor:
    def __init__(self,model_path):
        self.model = keras.models.load_model(model_path)
    def predict(self,patient):
        to_predict = np.expand_dims(patient.data,axis=0)
        x = self.model.predict(to_predict)
        return x

x = get_training_set("data\\class2.json","data\\npy\\")
pred = Predictor("web/classifier/final.h5")
print(pred.predict(x[0]))