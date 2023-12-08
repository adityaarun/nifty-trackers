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

# Start Spark session
spark = SparkSession.builder.appName("StockMarketPrediction") \
.config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1") \
.getOrCreate()

query = (
    spark.readStream.format("mongodb")
        .option('spark.mongodb.connection.uri', 'mongodb://localhost:27017')
        .option('spark.mongodb.database', 'iisc') \
        .option('spark.mongodb.collection', 'stock-actual') \
        .option('spark.mongodb.change.stream.publish.full.document.only','true') \
        .option("forceDeleteTempCheckpointLocation", "true") \
        .load()
)

query.printSchema()

logging.warn("Query is streaming: {}".format(query.isStreaming))

query2 = (
    query.writeStream \
        .outputMode("append") \
        .option("forceDeleteTempCheckpointLocation", "true") \
        .format("console") \
        .trigger(continuous="1 second")
        .start().awaitTermination()
)