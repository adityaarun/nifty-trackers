from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.pipeline import PipelineModel

import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import pyspark.sql.types as tp
from pyspark.sql.types import StringType, BooleanType, IntegerType, FloatType, DateType, Row
from datetime import datetime, timedelta
import logging

def train(spark, stock):
    # Start Spark session
    
    stocks_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'stock-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    index_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'index-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    commodities_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'commodity-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    news_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'news-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    
    start_timestamp = int((datetime.today() - timedelta(days = 365)).timestamp()) * 1000
    end_timestamp = int((datetime.today() - timedelta(days = 1)).timestamp()) * 1000
    logging.warn("Stock: {}, Start: {}, End: {}".format(stock, start_timestamp, end_timestamp))
    
    df = stocks_data
    df = df.filter((F.col('Symbol') == stock) & (F.col('Date') >= start_timestamp) & (F.col('Date') <= end_timestamp))
    logging.warn("DF Count: {}".format(df.count()))
    
    news_data = news_data.filter((F.col('Symbol') == stock) & (F.col('Date') >= start_timestamp) & (F.col('Date') <= end_timestamp)).select('Date', 'Sentiment')
    logging.warn("News DF Count: {}".format(news_data.count()))

    # Window function to get the previous day price of stock
    w = Window.partitionBy("Symbol").orderBy("Date")
    
    # 1. Get the previous day close price
    df = df.withColumn("prev_close", F.lag(df['Close']).over(w))

    #Calculate the stock percentage change from previous day close to today's close
    df = df.withColumn("change", (df['Close'] - df.prev_close) / df.prev_close)

    # 3. Calculate present day volatility by getting difference of high and low price of stock
    df = df.withColumn("day_volatility", df['High'] - df['Low'])

    # 4. Calculate volatility of stock by getting diff of close and open
    df = df.withColumn("daily_volatility", df['Close'] - df['Open'])

    # 5. getting the weekly moving average
    df = df.withColumn("weeklyMA", F.avg(df['Close']).over(w.rowsBetween(-6, 0)))

    # Drop any rows with NA values (which might have been introduced due to lagging operations)
    df = df.dropna()

    if news_data.count() > 0:
        news_data = news_data.groupBy('Date').agg(F.sum('Sentiment').alias('Sentiment'))
        news_data.show()
        df = df.join(news_data, df.Date == news_data.Date, "left").select(df["*"], news_data[news_data.columns[1]])
        df = df.fillna(0)
    else:
        df = df.withColumn("Sentiment", F.lit(0))

    train_data, test_data = df.randomSplit([0.75,0.25], seed = 42)

    #Code for feature assemble
    feature_list = ['Open', 'High', 'Low', 'Volume', "change", "day_volatility", "daily_volatility", "weeklyMA", "Sentiment"]
    stage_1 = VectorAssembler(inputCols=feature_list, outputCol="features")
    model = GBTRegressor(labelCol='Close', featuresCol="features", maxIter=50)
    # model = LinearRegression(featuresCol = 'features', labelCol = 'Close')

    pipeline = Pipeline(stages= [stage_1, model])
    pipeline_fit = pipeline.fit(train_data)
    predictions = pipeline_fit.transform(test_data)
    
    pipeline_fit.write().overwrite().save("/home/adarun/trained_models/"+stock)