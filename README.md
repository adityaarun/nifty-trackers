# NIFTY TRACKER
A spark based modeling tool for tracking stocks and nifty index.

## Organization

Code is organized in 3 folders.
1. data-fetch - This folder contains the code to pull data from various sources and insert them into MongoDB.
2. flask - This folder contains the code to pull data from Mongo to PySpark. It uses that data to train a model and store it in filesystem. The system also exposes a rudimentary flask UI to control training and get predictions.
3. model-trainer - This folder contains the streaming logic to detect changes in Mongo collection and trigger model retraining.
