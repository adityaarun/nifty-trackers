# NIFTY TRACKER
A spark based modeling tool for tracking stocks and nifty index. The source contains the methods to pull data from yfinance and store them in a MongoDB.
There is a streaming program which keeps track of changes in the stock-actual collection and trains new models for the symbols for which changes have happened.
A flask UI gives a basic command and control and prediction capabilities.

## Organization

Code is organized in 3 folders.
1. data-fetch - This folder contains the code to pull data from various sources and insert them into MongoDB.
2. flask - This folder contains the code to pull data from Mongo to PySpark. It uses that data to train a model and store it in filesystem. The system also exposes a rudimentary flask UI to control training and get predictions.
3. model-trainer - This folder contains the streaming logic to detect changes in Mongo collection and trigger model retraining.

## Infra Requirements (tested against)
1. Ubuntu 22.04 (8 core, 16 GB memory) virtual machine
2. MongoDB instance (if there is no external instance, we can run a MongoDB in the same VM where we are going to run the code)

## Install dependencies
`pip3 install -r requirements.txt`

## Install MongoDB (in case no access to existing one)
### Install the server
`sudo apt install mongodb`

### Configure replica set
Replica set is required for streaming data using PySpark from MongoDB.

`vim /etc/mongod.conf`

Add the following lines in replication section and save.

```
replication:
  replSetName: rs0
```

Run the following commands in mongo shell
```
use admin
rs.initiate()
exit
```

Restart the MongoDB server
```
sudo systemctl restart mongod.service
```

## Setup crons for pulling data
```
0 10 * * * /usr/bin/python3 /nifty_trackers/data-fetch/get_nifty_50.py
0 8 * * * /usr/bin/python3 /nifty_trackers/data-fetch/get_commodities.py
0 6 * * * /usr/bin/python3 /nifty_trackers/data-fetch/get_nifty.py
0 4 * * * /usr/bin/python3 /nifty_trackers/data-fetch/sentiment.py
```

## Start the streamer
```
cd /nifty_trackers/model-trainer
python3 streamer.py
```

## Start the flask app
```
cd /nifty_trackers/flask
python3 app.py
```