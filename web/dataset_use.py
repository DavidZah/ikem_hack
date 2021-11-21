import pickle
from pathlib import Path

from tensorflow import keras
import tensorflow
import numpy as np

from nlp_pdf.nlp_pdf_main import complete_model

def load_weights_by_name(model, path, verbose=False):
    import h5py
    def load_model_weights(cmodel, weights):
        for layer in cmodel.layers:
            print(layer.name)
            if hasattr(layer, 'layers'):
                load_model_weights(layer, weights[layer.name])
            else:
                for w in layer.weights:
                    _, name = w.name.split('/')
                    if verbose:
                        print(w.name)
                    try:
                        w.assign(weights[layer.name][name][()])
                    except:
                        w.assign(weights[layer.name][layer.name][name][()])


class Predictor:
    def __init__(self,xml_model_path,pdf_model_path,vectorizer_path,nlp_pdf_weights):
        self.xml_model = keras.models.load_model(xml_model_path)
        self.pdf_model = keras.models.load_model(pdf_model_path)
        self.pdf_nlp_model = complete_model()
        load_weights_by_name(self.pdf_nlp_model,nlp_pdf_weights,True)
        self.vectorizer = self.load_vectoriter(vectorizer_path)

    def load_pickle(self,path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

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
            string = patient.nlp
            # string = string.decode("utf-8")
            vec = self.vectorizer.transform([string]).toarray()
            x_1 = np.expand_dims(vec, axis=0)

            data = np.ndarray.transpose(patient.data)

            x_2 = np.expand_dims(data, axis=0)
            y =self.pdf_nlp_model.predict((x_1,x_2))
            return y[0][0][0]


if __name__ == "__main__":
    pred = Predictor("final.h5")
