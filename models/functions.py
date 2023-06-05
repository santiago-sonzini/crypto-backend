# 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
import pickle



# 
def getdata(coin): 
    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={coin}&market=ARS&apikey=P7226BKG9ND08BME'
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient='index')
    df = df.drop(['1a. open (ARS)', '2a. high (ARS)', '3a. low (ARS)', '4a. close (ARS)', '6. market cap (USD)'], axis=1)
    
    df = df.rename(columns={"1b. open (USD)": "Open", "2b. high (USD)": "High", "3b. low (USD)": 'Low', '4b. close (USD)': 'Close', '5. volume': "Volume"})
    df.index = pd.to_datetime(df.index)
    return df

#
def prepare_data(data):
    required_features = ['Open', 'High', 'Low', 'Volume']
    output_label = 'Close'
    data = data.apply(lambda x: x.astype('float64'))
    train_data = data[60:900]
    x_train, x_test, y_train, y_test = train_test_split(
    train_data[required_features],
    train_data[output_label],
    test_size = 0.3)
    return x_train, x_test, y_train, y_test

def create_model(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("model: ",  model)
    print("model scored: ",  model.score(x_test, y_test))
    return model

# 
def get_models(coins):
    models = []
    for i in coins:
        df = getdata(i)
        x_train, x_test, y_train, y_test = prepare_data(df)
        model = create_model( x_train, x_test, y_train, y_test)
        save_model(model,i)
        models.append(model)
    return models
    
def save_model(model,coin):
    filename = f'{coin}LinearModel.sav'
    pickle.dump(model, open(filename, 'wb'))
    
def load_model(coin):
    model = pickle.load(open(f'{coin}LinearModel.sav', 'rb'))                 
    return model
    

#  
def get_today(coin, model):
    
    df = getdata(coin)
    
    today =df.loc['2023-06-05'].drop("Close").to_numpy().reshape(1,-1)
    
    prediction = model.predict(today).astype('float32')
    return prediction



# 
def plot_diff(x,y,prediction):
    plt.rcParams.update({'font.size': 7})
    plt.title("BTC linear regression")
    plt.plot(x,y,  label='Actual Price', )
    plt.plot(x,prediction, label='Predicted Price')
    plt.legend(loc="upper right")
    plt.xlabel("Date")
    plt.ylabel('Price (USD)')
    plt.show()


