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
from . import collect_data as libs
from datetime import datetime, timedelta
import logging

def train_nifty(start, end):
    spark = SparkSession.builder.appName("StockMarketPrediction") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1") \
    .getOrCreate()
    
    stocks_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'stock-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    index_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'index-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    commodities_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'commodity-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    start_timestamp = int(datetime.strptime(start,"%Y-%m-%d").timestamp()) * 1000
    end_timestamp = int(datetime.strptime(end,"%Y-%m-%d").timestamp()) * 1000
    stocks_data = stocks_data.filter((F.col('Date') >= start_timestamp) & (F.col('Date') <= end_timestamp))
    # Window function to get the previous day price of stock
    w = Window.partitionBy("Symbol").orderBy("Date")
    
    # 1. Get the previous day close price
    stocks_data = stocks_data.withColumn("prev_close", F.lag(stocks_data['Close']).over(w))

    #Calculate the stock percentage change from previous day close to today's close
    stocks_data = stocks_data.withColumn("change", (stocks_data['Close'] - stocks_data.prev_close) / stocks_data.prev_close)

    # 3. Calculate present day volatility by getting difference of high and low price of stock
    stocks_data = stocks_data.withColumn("day_volatility", stocks_data['High'] - stocks_data['Low'])

    # 4. Calculate volatility of stock by getting diff of close and open
    stocks_data = stocks_data.withColumn("daily_volatility", stocks_data['Close'] - stocks_data['Open'])

    # 5. getting the weekly moving average
    stocks_data = stocks_data.withColumn("weeklyMA", F.avg(stocks_data['Close']).over(w.rowsBetween(-6, 0)))

    # Drop any rows with NA values (which might have been introduced due to lagging operations)
    stocks_data = stocks_data.fillna(0).select("Date", "Symbol", "Close")

    stocks_data.show()
    symbols = [list(x.asDict().values())[0] for x in stocks_data.select("Symbol").distinct().collect()]
    dfArray = [stocks_data.where(stocks_data.Symbol == x).withColumn(x + "_Close", stocks_data.Close).drop("Symbol", "Close") for x in symbols]
    
    for df in dfArray:
        index_data = index_data.join(df, index_data.Date == df.Date, "left").select(index_data["*"], df[df.columns[1]])

    index_data = index_data.fillna(0)
    
    feature_list = index_data.columns
    feature_list.remove("Close")
    feature_list.remove("_id")
    feature_list.remove("Symbol")
    logging.warn(feature_list)
    
    train_data, test_data = index_data.randomSplit([0.75,0.25], seed = 42)
    stage_1 = VectorAssembler(inputCols=feature_list, outputCol="features")
    model = GBTRegressor(labelCol='Close', featuresCol="features", maxIter=50)
    # model = LinearRegression(featuresCol = 'features', labelCol = 'Close')

    pipeline = Pipeline(stages= [stage_1, model])
    pipeline_fit = pipeline.fit(train_data)
    predictions = pipeline_fit.transform(test_data)
    
    pipeline_fit.write().overwrite().save("/home/adarun/trained_models/NIFTY")

    #Model Evaluation
    #Calculate evaluation metric like RMSE and R2
    evaluator_rmse = RegressionEvaluator(labelCol='Close', predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    logging.warn("RMSE on test_data is = {}".format(rmse))

    # Compute other metrics: Mean Absolute Error (MAE) and R-squared (R2)
    for metric in ["mae", "r2"]:
        evaluator = RegressionEvaluator(labelCol='Close', predictionCol="prediction", metricName=metric)
        value = evaluator.evaluate(predictions)
        logging.warn(f"{metric.upper()}: {value}")

    preds = predictions.withColumn('Date', F.from_unixtime(F.col('Date') / 1000, "yyyy-MM-dd HH:mm:ss")).select("Date", "Close", "prediction").toPandas()  

    spark.stop()

    return preds

def train(stock, start, end):
    # Start Spark session
    spark = SparkSession.builder.appName("StockMarketPrediction") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1") \
    .getOrCreate()
    
    stocks_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'stock-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    index_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'index-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    commodities_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'commodity-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    news_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'news-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    
    start_timestamp = int(datetime.strptime(start,"%Y-%m-%d").timestamp()) * 1000
    end_timestamp = int(datetime.strptime(end,"%Y-%m-%d").timestamp()) * 1000
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

    #Model Evaluation
    #Calculate evaluation metric like RMSE and R2
    evaluator_rmse = RegressionEvaluator(labelCol='Close', predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    logging.warn("RMSE on test_data is = {}".format(rmse))

    # Compute other metrics: Mean Absolute Error (MAE) and R-squared (R2)
    for metric in ["mae", "r2"]:
        evaluator = RegressionEvaluator(labelCol='Close', predictionCol="prediction", metricName=metric)
        value = evaluator.evaluate(predictions)
        logging.warn(f"{metric.upper()}: {value}")

    preds = predictions.withColumn('Date', F.from_unixtime(F.col('Date') / 1000, "yyyy-MM-dd HH:mm:ss")).select("Date", "Close", "prediction").toPandas()    
    # close the Spark session
    spark.stop()

    return preds

def prediction(stock, date, open, high, low, volume, change, day_volatility, daily_volatility, weeklyMA):
    # Start Spark session
    spark = SparkSession.builder.appName("StockMarketPrediction") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1") \
    .getOrCreate()
    
    stocks_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'stock-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    index_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'index-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    commodities_data = spark.read.format("mongodb").option("spark.mongodb.read.database", 'iisc').option("spark.mongodb.read.collection", 'commodity-actual').option("spark.mongodb.write.connection.uri","mongodb://localhost:27017").load()
    
    persistedModel = PipelineModel.load("/home/adarun/trained_models/"+stock)

    predict_df = spark.createDataFrame([
        Row(
            Date=date, Close=0, Open = float(open), 
            High = float(high), Low = float(low), Volume = float(volume), 
            change = float(change), day_volatility = float(day_volatility), daily_volatility = float(daily_volatility), 
            weeklyMA = float(weeklyMA), Sentiment = 0
        )
    ])
    predictions = persistedModel.transform(predict_df).select("Date", "prediction")
    predictions.show()
    preds = predictions.withColumn('Date', F.from_unixtime(F.col('Date') / 1000, "yyyy-MM-dd HH:mm:ss")).select("Date", "prediction").toPandas()    
    # close the Spark session
    spark.stop()

    return preds