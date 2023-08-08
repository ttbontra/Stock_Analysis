import numpy as np
import tensorflow as tf
from time import time
from sklearn.preprocessing import MinMaxScaler

def train_model(stock_data, epochs=50, batch_size=34):
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(stock_data)

    features = []
    labels = []
    for i in range(60, len(stock_data)):
        features.append(training_set_scaled[i - 60:i, 0])
        labels.append(training_set_scaled[i, 0])

    features = np.array(features)
    labels = np.array(labels)

    features = np.reshape(features, (features.shape[0], features.shape[1], 1))

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

    model.compile(optimizer='adam', loss='mean_squared_error')

    start = time()
    history = model.fit(features, labels, epochs=epochs, batch_size=batch_size, verbose=1)
    end = time()
    print('Total training time {} seconds'.format(end - start))

    model.save('stock_prediction.keras')

    return model, sc

def forecast(model, input_data, num_prediction):
    prediction_list = input_data[-1].reshape(-1)  # Take the last sequence from input_data
    
    for _ in range(num_prediction):
        x = prediction_list[-60:]
        x = x.reshape((1, 60, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[60-1:]
        
    return prediction_list
