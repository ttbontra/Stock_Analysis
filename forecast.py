import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
from time import time
from keras.models import load_model
from get_data import fetch_data

def train_and_forecast(ticker_symbol):
    model = load_model('stock_prediction.keras')
    stock_data = fetch_data(ticker_symbol, timeframe='120mo')
    # Specify the path to your CSV file
    if stock_data is None:
        return None, None, None, None

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
    #plt.figure(figsize=(10, 6))
    #plt.plot(test_stock_data_processed, color='blue', label='Actual Apple Stock Price')
    #plt.plot(predicted_stock_price, color='red', label='Predicted Apple Stock Price')
    #plt.title('SPY Stock Price Prediction')
    #plt.xlabel('Date')
    #plt.ylabel('SPY Stock Price')
    #plt.legend()
    #plt.show()
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
    print(predicted_stock_price.shape)
    print(test_stock_data_processed.shape)
    x_actual = np.arange(test_stock_data_processed.shape[0])
    x_predicted = np.arange(test_stock_data_processed.shape[0], test_stock_data_processed.shape[0] + predicted_stock_price.shape[0])
    #plt.figure(figsize=(10, 6))
    #plt.plot(x_actual, test_stock_data_processed, color='blue', label=f'Actual {tickers} Stock Price')
    #plt.plot(x_predicted, predicted_stock_price, color='red', label=f'Predicted {tickers} Stock Price')
    #plt.title(f'{tickers} Stock Price Prediction')
    #plt.xlabel('Date')
    #plt.ylabel(f'{tickers} Stock Price')
    #plt.legend()
    #plt.show()
    return x_actual, test_stock_data_processed, x_predicted, predicted_stock_price