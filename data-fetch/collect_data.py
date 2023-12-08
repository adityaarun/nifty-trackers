import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from pyspark.sql import SparkSession
import requests

#Get nifty50 stocks list 
def get_nifty50_stock_list():
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract the Nifty 50 stock symbols from the HTML
        stock_list = [line.split(",")[2] for line in soup.text.splitlines()][1:]

        return stock_list
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_stock_data(stock, start=None, end=None):
    try:
        if stock == 'NIFTY':
            stock = '^NSEI'
        elif stock == 'CRUD':
            stock = 'CL=F'
        elif stock == 'GOLD':
            stock = 'GC=F'
        else:
            stock = stock+'.NS'
        if start and end:
            data = yf.download(stock, start = start , end = end)
        else:
            data = yf.download(stock, start = start , end = datetime.now())

        data = data.drop(['Adj Close'], axis=1)
        data.reset_index(inplace=True)

        return data
    except Exception as e:
        print(f"Error: {e}")
        return None
