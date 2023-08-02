import numpy as np
import pandas as pd
import math

class TradingSignals:
    def __init__(self, data):
        self.data = data
        self.short_window = 50
        self.long_window = 200

    def calculate_moving_averages(self):
        self.data['short_mavg'] = self.data['Close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        self.data['long_mavg'] = self.data['Close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

    def generate_signals(self):
        self.calculate_moving_averages()
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0

        # Create signals
        signals['signal'][self.short_window:] = np.where(self.data['short_mavg'][self.short_window:] 
            > self.data['long_mavg'][self.short_window:], 1.0, 0.0)   

        # Generate trading orders
        signals['positions'] = signals['signal'].diff()

        return signals
