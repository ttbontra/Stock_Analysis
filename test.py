import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
from time import time
from keras.models import load_model
from get_data import fetch_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_DATABASE'),
    'raise_on_warnings': True
}

def insert_into_db(ticker, forecast_date, predicted_price, model_name, slope=None):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    insert_query = """
    INSERT INTO stock_forecast (ticker, date, predicted_price, model_name, slope)
    VALUES (%s, %s, %s, %s, %s);
    """

    try:
        cursor.execute(insert_query, (ticker, forecast_date, predicted_price, model_name, slope))
        cnx.commit()
    except mysql.connector.Error as err:
        print(f"Error inserting prediction into stock_forecast: {err}")

    cursor.close()
    cnx.close()

# ---------------- Data Preparation ------------------

def fetch_and_prepare_data(ticker_symbol, timeframe='120mo'):
    stock_data = fetch_data(ticker_symbol, timeframe)
    if stock_data is None:
        return None

    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    start_date = stock_data['Date'].min().tz_localize(None)
    end_date = stock_data['Date'].max().tz_localize(None)

    all_bussinessdays = pd.date_range(start=start_date, end=end_date, freq='B')
    close_prices = stock_data.reindex(all_bussinessdays)
    close_prices = stock_data.fillna(method='ffill')
    return close_prices

# ---------------- Model Management ------------------

def create_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, features, labels, epochs=50, batch_size=34):
    history = model.fit(features, labels, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save('stock_prediction.keras')

# ---------------- Predictions ------------------

def make_predictions(model, test_features, num_prediction):
    predicted_stock_price = model.predict(test_features)
    predicted_stock_price = predict(num_prediction, model, test_features)
    return predicted_stock_price

# Function predict remains as it is in your code

# ---------------- Database Operations ------------------

def save_predictions_to_db(ticker_symbol, predicted_stock_price, test_stock_data):
    slope = float((predicted_stock_price[-1] - predicted_stock_price[0]) / len(predicted_stock_price))
    start_date = test_stock_data['Date'].iloc[-1]
    forecast_dates = pd.bdate_range(start=start_date, periods=num_prediction+1)[1:]

    for i in range(len(predicted_stock_price)):
        forecast_date = forecast_dates[i].strftime('%Y-%m-%d')
        rounded_price = round(float(predicted_stock_price[i]), 4)
        insert_into_db(ticker_symbol, forecast_date, float(predicted_stock_price[i]), 'NeuralNetwork', slope)
        print(f"Inserting: Date={forecast_date}, Predicted Price={rounded_price}, Slope={slope}")

# ---------------- Main Function ------------------

def train_and_forecast(ticker_symbol):
    close_prices = fetch_and_prepare_data(ticker_symbol)
    if close_prices is None:
        return None, None, None, None

    # Rest of the data preparation, creating training and testing datasets...

    # Load existing model or create a new one
    try:
        model = load_model('stock_prediction.keras')
    except:
        input_shape = (features.shape[1], 1)
        model = create_lstm_model(input_shape)

    train_lstm_model(model, features, labels)

    predicted_stock_price = make_predictions(model, test_features, num_prediction)

    save_predictions_to_db(ticker_symbol, predicted_stock_price, test_stock_data)

    x_actual = np.arange(test_stock_data_processed.shape[0])
    x_predicted = np.arange(test_stock_data_processed.shape[0], test_stock_data_processed.shape[0] + predicted_stock_price.shape[0])
    return x_actual, test_stock_data_processed, x_predicted, predicted_stock_price
