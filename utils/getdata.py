import requests
import pandas as pd
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=ETH&market=ARS&apikey=P7226BKG9ND08BME'


def getdata(): 
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient='index')
    df = df.drop(['1a. open (ARS)', '2a. high (ARS)', '3a. low (ARS)', '4a. close (ARS)', '6. market cap (USD)'], axis=1)
    df = df.rename(columns={"1b. open (USD)": "Open", "2b. high (USD)": "High", "3b. low (USD)": 'Low', '4b. close (USD)': 'Close', '5. volume': "Volume"})
    return df
