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



config = {
    'user': 'root',
    'password': 'stable',
    'host': '127.0.0.1',
    'database': 'stocks',
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

def save_predictions_to_db(ticker_symbol, predicted_stock_price, test_stock_data):
    slope = float((predicted_stock_price[-1] - predicted_stock_price[0]) / len(predicted_stock_price))
    start_date = test_stock_data['Date'].iloc[-1]
    forecast_dates = pd.bdate_range(start=start_date, periods=num_prediction+1)[1:]

    for i in range(len(predicted_stock_price)):
        forecast_date = forecast_dates[i].strftime('%Y-%m-%d')
        rounded_price = round(float(predicted_stock_price[i]), 4)
        insert_into_db(ticker_symbol, forecast_date, float(predicted_stock_price[i]), 'NeuralNetwork', slope)
        print(f"Inserting: Date={forecast_date}, Predicted Price={rounded_price}, Slope={slope}")


def forecast_with_model(train, test, model_name):
    if model_name == "LinearRegression":
        return linear_regression_forecast(train, test)
    elif model_name == "ARIMA":
        return arima_forecast(train, test)
    elif model_name == "RandomForest":
        return random_forest_forecast(train, test)
    elif model_name == "XGBoost":
        return xgboost_forecast(train, test)
    elif model_name == "NeuralNetwork":
        # The LSTM neural network model prediction code
        pass
    else:
        print(f"Unknown model: {model_name}")
        return None

def visualize_results(x_actual, actual_prices, model_results):
    plt.figure(figsize=(14, 7))
    plt.plot(x_actual, actual_prices, color="black", label="Actual Prices")
    
    for model_name, x_predicted, predicted_prices in model_results:
        plt.plot(x_predicted, predicted_prices, label=f"Predicted by {model_name}")

    plt.title("Stock Price Forecasting Comparison")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

def linear_regression_forecast(train, test):
    regressor = LinearRegression()
    regressor.fit(np.arange(len(train)).reshape(-1, 1), train)
    predictions = regressor.predict(np.arange(len(train), len(train)+len(test)).reshape(-1, 1))
    return predictions

def arima_forecast(train, test):
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    predictions, _, _ = model_fit.forecast(steps=len(test))
    return predictions

def random_forest_forecast(train, test):
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(np.arange(len(train)).reshape(-1, 1), train)
    predictions = regressor.predict(np.arange(len(train), len(train)+len(test)).reshape(-1, 1))
    return predictions

def xgboost_forecast(train, test):
    train_data = xgb.DMatrix(np.arange(len(train)).reshape(-1, 1), label=train)
    test_data = xgb.DMatrix(np.arange(len(train), len(train)+len(test)).reshape(-1, 1))
    param = {'max_depth': 5, 'eta': 0.1, 'objective': 'reg:squarederror'}
    num_round = 100
    bst = xgb.train(param, train_data, num_round)
    predictions = bst.predict(test_data)
    return predictions

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

def train_and_forecast(ticker_symbol):
    model = load_model('stock_prediction.keras')
    stock_data = fetch_data(ticker_symbol, timeframe='120mo')
    # Specify the path to your CSV file
    if stock_data is None:
        return None, None, None, None
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Extract the start date (the earliest date in the 'Date' column)
    start_date = stock_data['Date'].min()
    # Extract the end date (the latest date in the 'Date' column)
    end_date = stock_data['Date'].max()

    stock_data_len = stock_data['Close'].count()
    close_prices = stock_data.iloc[:, 1:2].values
    print(close_prices)
    # Convert the start and end dates to be timezone naive
    start_date = start_date.tz_localize(None)
    end_date = end_date.tz_localize(None)
    all_bussinessdays = pd.date_range(start=start_date, end=end_date, freq='B')
    print(all_bussinessdays)
    close_prices = stock_data.reindex(all_bussinessdays)
    close_prices = stock_data.fillna(method='ffill')
    close_prices.head(10)
    training_set = close_prices.iloc[:, 1:2].values
    print(training_set)
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    print(training_set_scaled.shape)
    features = []
    labels = []
    for i in range(60, stock_data_len):
        features.append(training_set_scaled[i - 60:i, 0])
        labels.append(training_set_scaled[i, 0])
    features = np.array(features)
    labels = np.array(labels)
    features = np.reshape(features, (features.shape[0], features.shape[1], 1))
    print(labels)
    print(features)
    print(features.shape)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(features.shape[1], 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1)
    ])
    print(model.summary())
    model.compile(optimizer='adam', loss='mean_squared_error')
    start = time()
    history = model.fit(features, labels, epochs=50, batch_size=34, verbose=1) #change features to generate_batches(files, batch_size)
    model.save('stock_prediction.keras')
    end = time()
    print('Total training time {} seconds'.format(end - start))
    print(features.shape)
    testing_start_date = pd.to_datetime('2023-04-15').tz_localize('US/Eastern')
    testing_end_date = pd.to_datetime('2023-07-12').tz_localize('US/Eastern')
    test_stock_data = stock_data[(stock_data['Date'] >= testing_start_date) & (stock_data['Date'] <= testing_end_date)]
    test_stock_data.tail()
    test_stock_data_processed = test_stock_data.iloc[:, 1:2].values
    print(test_stock_data_processed.shape)
    all_stock_data = pd.concat((stock_data['Close'], test_stock_data['Close']), axis=1)
    inputs = all_stock_data[len(all_stock_data) - len(test_stock_data) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 129):
        if len(inputs[i - 60:i, 0]) == 60:
            X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    num_prediction = 30  # Number of future predictions
    

    def predict(num_prediction, model, input_data):
        prediction_list = input_data[-1].reshape(-1)  # Take the last sequence from input_data
        
        for _ in range(num_prediction):
            x = prediction_list[-60:]
            x = x.reshape((1, 60, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[60-1:]
            
        return prediction_list
    test_inputs = test_stock_data_processed.reshape(-1, 1)
    test_inputs = sc.transform(test_inputs)
    print(test_inputs.shape)
    test_features = []
    for i in range(60, len(test_inputs) + 1):
        if len(test_inputs[i - 60:i, 0]) == 60:
            test_features.append(test_inputs[i - 60:i, 0])
    test_features = np.array(test_features)
    print("Shape of test_features before reshaping:", test_features.shape)

    # Then proceed with reshaping if it has more than 1 dimension
    if len(test_features.shape) > 1:
        test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
        print("Shape of test_features after reshaping:", test_features.shape)
    else:
        print("Cannot reshape, test_features is not a 2D array.")
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    print(model.summary())
    #predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price = model.predict(test_features)
    predicted_stock_price = predict(num_prediction, model, test_features)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price.reshape(-1, 1))
    slope = round(float((predicted_stock_price[-1] - predicted_stock_price[0]) / len(predicted_stock_price)), 4)
    start_date = test_stock_data['Date'].iloc[-1]
    forecast_dates = pd.bdate_range(start=start_date, periods=num_prediction+1)[1:]
    

    for i in range(num_prediction):
        forecast_date = forecast_dates[i].strftime('%Y-%m-%d')
        rounded_price = round(float(predicted_stock_price[i]), 4)
        insert_into_db(ticker_symbol, forecast_date, rounded_price, 'NeuralNetwork', slope)
        print(f"Inserting: Date={forecast_date}, Predicted Price={rounded_price}, Slope={slope}")
    
    print(predicted_stock_price.shape)
    print(test_stock_data_processed.shape)
    x_actual = np.arange(test_stock_data_processed.shape[0])
    x_predicted = np.arange(test_stock_data_processed.shape[0], test_stock_data_processed.shape[0] + predicted_stock_price.shape[0])
    print(f"Inserting: Date={forecast_date}, Predicted Price={rounded_price}, Slope={slope}")

    return x_actual, test_stock_data_processed, x_predicted, predicted_stock_price

def make_predictions(model, test_features, num_prediction):
    predicted_stock_price = model.predict(test_features)
    predicted_stock_price = predict(num_prediction, model, test_features)
    return predicted_stock_price