import numpy as np
import tensorflow as tf
from tensorflow import keras
from kerastuner import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    
    # Tune the number of units in the first LSTM layer
    # Choose an optimal value between 32-256
    model.add(keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                                return_sequences=True,
                                input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.Dropout(0.2))
    
    # Tune the number of units in the second LSTM layer
    model.add(keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32)))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.Dense(units=1))
    
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse')
    
    return model


