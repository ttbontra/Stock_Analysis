import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import plotly.graph_objects as go
from time import time
import pydot
import graphviz
from keras.models import load_model
import math
from datetime import datetime
import datetime as dt    
sns.set_style("whitegrid")
from pandas.plotting import autocorrelation_plot
from keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from keras.layers import MaxPooling1D, Flatten
from keras.regularizers import L1, L2
from keras.metrics import Accuracy
from keras.metrics import RootMeanSquaredError
from keras.utils import plot_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import explained_variance_score, mean_poisson_deviance, mean_gamma_deviance
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QProgressBar, QLabel, QTabWidget
from PyQt5.QtCore import QUrl, QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebEngineWidgets import QWebEnginePage

# Load and preprocess the data
data = pd.read_csv('SPY.csv')
data.dropna(inplace=True)
data['Daily Return'] = data['Close'].pct_change()

# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    # Daily Return plot
    dcc.Graph(
        id='daily-return',
        figure=px.line(data, x='Date', y='Daily Return', title='Daily Return', markers=True, line_shape='linear')
    ),
    # Histogram plot
    dcc.Graph(
        id='histogram',
        figure=px.histogram(data, x='Daily Return', nbins=100, color_discrete_sequence=['green'])
    ),
    # Multiple line plots for Open, High, Low, Close
    dcc.Graph(
        id='ohlc',
        figure={
            'data': [
                {'x': data['Date'], 'y': data[col], 'type': 'scatter', 'mode': 'markers', 'name': col} for col in ['Open', 'High', 'Low', 'Close']
            ],
            'layout': {
                'title': 'Open, High, Low, Close over Time'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)