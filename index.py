from flask import Flask
from models.functions import load_model, get_today
import json

app = Flask(__name__)



ETHModel = load_model("ETH")
BTCModel = load_model("BTC")
DOGEModel = load_model("DOGE")


@app.route('/eth')
def home():
    prediction = get_today("ETH", ETHModel)
    prediction = {'eth': prediction.tolist()}
    print("prediccion", prediction["eth"])
    return  json.dumps(prediction) 

@app.route('/btc')
def home():
    prediction = get_today("BTC", BTCModel)
    prediction = {'btc': prediction.tolist()}
    print("prediccion", prediction["btc"])
    return  json.dumps(prediction) 

@app.route('/doge')
def home():
    prediction = get_today("DOGE", DOGEModel)
    prediction = {'doge': prediction.tolist()}
    print("prediccion", prediction["doge"])
    return  json.dumps(prediction) 

if __name__ == '__main__':
    app.run(port=3000,debug=True)