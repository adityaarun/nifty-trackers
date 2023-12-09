# app.py

from flask import Flask, render_template, request
from lib import collect_data as lib
from lib import Prediction as pred
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

def display_welcome_message():
    return "Welcome! Input processed successfully."

@app.route('/')
def index():
    stock_lst = lib.get_nifty50_stock_list()
    stock_list = stock_lst # + ['NIFTY', 'CRUD']
    return render_template('index.html', stock_lst=stock_list)

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'GET':
        stock_lst = lib.get_nifty50_stock_list()
        stock_list = stock_lst + ['NIFTY', 'CRUD']
        return render_template('process.html', stock_lst=stock_list)
    
    text_input = request.form['text_input']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    predictions =  pred.train(text_input, start_date, end_date)

    print("Predictions")
    print(predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(predictions["Date"], predictions['Close'], label='Actual', color='blue')
    plt.plot(predictions["Date"], predictions["prediction"], label='Predicted', color='red', alpha=0.6)
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the plot image to base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Pass the base64-encoded image to the HTML template
    return render_template('train_output.html', img_base64=img_base64)

@app.route('/process_index', methods=['GET', 'POST'])
def process_index():
    if request.method == 'GET':
        return render_template('process_index.html')
    
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    predictions =  pred.train_nifty(start_date, end_date)

    print("Predictions")
    print(predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(predictions["Date"], predictions['Close'], label='Actual', color='blue')
    plt.plot(predictions["Date"], predictions["prediction"], label='Predicted', color='red', alpha=0.6)
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the plot image to base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Pass the base64-encoded image to the HTML template
    return render_template('train_output.html', img_base64=img_base64)

@app.route('/predict', methods=['POST'])
def predict():
    stock = request.form['stock']
    date = request.form['date']
    open = request.form['open']
    high = request.form['high']
    low = request.form['low']
    volume = request.form['volume']
    change = request.form['change']
    day_volatility = request.form['day_volatility']
    daily_volatility = request.form['daily_volatility']
    weeklyMA = request.form['weeklyMA']

    predictions =  pred.prediction(stock, date, open, high, low, volume, change, day_volatility, daily_volatility, weeklyMA)

    print("Predictions")
    print(predictions)

    # Pass the base64-encoded image to the HTML template
    return render_template('prediction.html', predictions=predictions["prediction"].values[0])

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'GET':
        stock_lst = lib.get_nifty50_stock_list()
        stock_list = stock_lst + ['NIFTY', 'CRUD']
        return render_template('compare.html', stock_lst=stock_list)
    
    stock1 = request.form['stock1']
    stock2 = request.form['stock2']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    stock1, stock2, compare_data =  pred.comparison(stock1, stock2, start_date, end_date)

    fig, ax1 = plt.subplots(figsize=(15, 6))
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.tick_params(axis = 'x', labelrotation=60)
    ax1.set_ylabel('Close', color=color)
    ax1.plot(compare_data["Date"], compare_data["CloseStock1"], color=color, label=stock1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Close', color=color)  # we already handled the x-label with ax1
    ax2.plot(compare_data["Date"], compare_data["CloseStock2"], color=color, label=stock2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('{} vs {} Closing Prices'.format(stock1, stock2))
    plt.legend()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the plot image to base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Pass the base64-encoded image to the HTML template
    return render_template('train_output.html', img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8000')
