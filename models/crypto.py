from tensorflow import keras
from utils.getdata import getdata
from sklearn.preprocessing import MinMaxScaler
import numpy as np

__path__ = "/model/model.keras"
model = keras.models.load_model('/Users/santiagosonzini/Downloads/crypto-main/model/model.keras', compile=False)
model.compile(optimizer="adam",loss="mean_squared_error")
scaler = MinMaxScaler(feature_range=(0,1))

data = getdata()
data = data[:60]
data  = data.reshape(-1,1)
data = scaler.fit_transform(data)

data = np.array(data)
data = np.reshape(data, (data.shape[0], data.shape[1],1))


def get_prediction(data):
    prediction=model.predict(data)
    return prediction

