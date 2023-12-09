import numpy as np
import matplotlib.pyplot as plt
import string
import re
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, time
import requests
import traceback
import pdb
import os
import json
from functools import reduce
import pandas as pd

import feedparser

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, from_unixtime, unix_timestamp,col , lit, date_format
from pyspark.sql.types import StringType, DateType, FloatType

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pymongo import MongoClient

nltk.download('vader_lexicon')

def extract_value(summary, exchange):
    pattern = re.compile(f"{exchange}[^0-9]*([0-9,.]+)")
    match = pattern.search(summary)
    if match:
        value_str = match.group(1).replace(',', '')
        return float(value_str)
    return None

def extract_company_name(title):
    # Assuming the company name is at the beginning of the title
    match = re.match(r"([^\d\W]+)", title)
    if match:
        return match.group(1)
    return None

def analyze_sentiment(title, sia):
    sentiment_score = sia.polarity_scores(title)
    compound_score = sentiment_score['compound']

    # Categorize the sentiment
    if compound_score > 0.05:
        return 1
    elif compound_score < -0.05:
        return -1
    else:
        return 0

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

def get_news_feed_for_stock(ticker):
    dfs = []

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'}

    ticker = ticker + '.NS'

    rssfeedurl = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US' % ticker
    data = feedparser.parse(rssfeedurl)
    if data is not None and data.entries:
        return data
    else:
        return None

def process_sentiment_analysis(spark, db_collection, start_date, end_date, ticker):
    print(f"Processing stock: {ticker}")

    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Sample DataFrame with multiple entries
    data = get_news_feed_for_stock(ticker)

    if data is not None and data.entries:
        entry_list = []
        for i in range(min(1000, len(data.entries))):
            entry_dict = {
                "Symbol": ticker,
                "News": data.entries[i].title,
                "Date": datetime.combine(datetime.strptime(data.entries[i].published, '%a, %d %b %Y %H:%M:%S %z'), time.min),
                "Sentiment": analyze_sentiment(data.entries[i].title,sia)
            }
            entry_list.append(entry_dict)

        df = spark.createDataFrame(entry_list)
        df = df.withColumn(
                'Date',
                date_format('Date', "yyyy-MM-dd HH:mm:ss"))
        pandas_df = df.toPandas()
        pandas_df['Date'] = pandas_df['Date'].astype("datetime64[ns]")

        json_data = json.loads(pandas_df.to_json(orient='records'))
        if len(json_data) > 0:
            db_collection.insert_many(json_data)

client = MongoClient('localhost', 27017)
stocks_db = client['iisc']
news_actual = stocks_db['news-actual']

spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

stocks = get_nifty50_stock_list()

today = datetime.today()
start_date_str = (today - timedelta(days = 2)).strftime("%Y-%m-%d")
end_date_str = (today - timedelta(days = 1)).strftime("%Y-%m-%d")

for stock in stocks:
    process_sentiment_analysis(spark, news_actual, start_date_str, end_date_str, stock)

spark.stop()
client.close()