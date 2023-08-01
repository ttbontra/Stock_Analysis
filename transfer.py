from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load your data
data = pd.read_csv('MSFT_history.csv')

# Add technical indicators
data['SMA'] = data['Close'].rolling(window=60).mean()

# Normalize your data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data[['Close', 'SMA']].dropna())

close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit_transform(data[['Close']])

# Split the data into a training set and a test set
train_size = int(len(data_normalized) * 0.8)
data_train = data_normalized[:train_size]
data_test = data_normalized[train_size:]

X = []
y = []
X_test = []
y_test = []

# Create sequences
for i in range(60, len(data_train)):
    X.append(data_train[i - 60:i])
    y.append(data_train[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Assume that 'data_test' is your test dataset
for i in range(60, len(data_test)):
    X_test.append(data_test[i - 60:i])
    y_test.append(data_test[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=10)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print('Test loss:', loss)

# Make predictions
predictions = model.predict(X_test)
predictions = close_scaler.inverse_transform(predictions)

# Define the number of future days to predict
future_days = 90

# Get the last window of data
last_window = data_normalized[-60:]

# Reshape it to match the input shape of the model
last_window = last_window.reshape((1, 60, 2)) #last_window.shape[0] pr 60


# Initialize an empty list to store the future predictions
future_predictions = []


# Loop over the future days
recent_closes = last_window[0, :, 0].tolist()

# Loop over the future days
for i in range(future_days):
    # Predict the next time step
    next_step = model.predict(last_window)
    next_step = next_step.reshape((1, 1, 1))

    # Append the prediction to the future predictions list
    future_predictions.append(next_step[0])

    # Add the prediction to the end of the recent_closes list and remove the oldest value
    recent_closes.append(next_step[0, 0, 0])
    recent_closes.pop(0)

    # Calculate the SMA as the average of the recent_closes
    sma = np.mean(recent_closes)

    # Add the prediction and SMA back into the last window of data
    last_window = np.append(last_window[:, 1:, :], [[next_step[0, 0, 0], sma]], axis=1)

# Inverse transform the predictions to get them back to the original scale
future_predictions = close_scaler.inverse_transform(future_predictions)

# Now 'future_predictions' contains the predicted close prices for the next 'future_days' days
# Create a range of future dates
future_dates = pd.date_range(start=data.index[-1], periods=future_days+1)[1:]

# Create a DataFrame with the future predictions
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Prediction'])

# Plot the original data
plt.figure(figsize=(14, 6))
plt.plot(data['Close'], label='Historical Close Price')

# Plot the future predictions
plt.plot(future_df['Prediction'], label='Future Predictions')

plt.title('Future Price Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.show()

# Create a new DataFrame for plotting
#plot_df = data[['Close']].iloc[train_size+1:][:len(predictions)-1]
#plot_df['Predictions'] = predictions[1:]

# Plot the actual close prices and the predicted prices
#plt.figure(figsize=(14, 8))
#plt.plot(plot_df['Close'], label='Actual Close Price')
#plt.plot(plot_df['Predictions'], label='Predicted Close Price')
#plt.title('Close Price vs Predicted Close Price')
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.legend()
#plt.show()
# Forecasting 90 days ahead
#forecast_period = 90
#forecast = []

# Use the last 60 days of data to forecast the next day, then add that prediction to the data and repeat
#current_batch = data_normalized[-60:].reshape((1, 60, data_normalized.shape[1]))
#for i in range(forecast_period):
#    current_pred = model.predict(current_batch)[0]
#    current_pred = current_pred.reshape((1, 1, 1))
#    forecast.append(current_pred)
#    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse transform the forecast to get it back to the original scale
#forecast = close_scaler.inverse_transform(forecast)

# Create a DataFrame for the forecast with corresponding future dates
#forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_period)
#forecast_df = pd.DataFrame(data=forecast, index=forecast_dates, columns=['Forecast'])

# Save signals to a CSV file
#predictions_df = pd.DataFrame(
#    {'Date': data.index[train_size + 1:][:len(predictions) - 1], 'Predictions': predictions[1:].flatten()})
#predictions_df = predictions_df.append(forecast_df.reset_index().rename(columns={'index': 'Date'}))
#predictions_df.to_csv('predictions.csv', index=False)
