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
from dash import dcc
from dash import html
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
dataX = pd.read_csv('SPY.csv')
dataY = pd.read_csv('SPY.csv')
dataX['Date'] = pd.to_datetime(dataX['Date'])
dataY['Date'] = pd.to_datetime(dataY['Date'])
# Box Plots and Distribution Plots for dataX
fig_dataX_close_box = px.box(dataX, y="Close", title="dataX Close Box Plot")
fig_dataX_close_hist = px.histogram(dataX, x="Close", title="dataX Close Distribution")
# ... Repeat for Open, High, Low

# Box Plots and Distribution Plots for dataY
fig_dataY_close_box = px.box(dataY, y="Close", title="dataY Close Box Plot")
fig_dataY_close_hist = px.histogram(dataY, x="Close", title="dataY Close Distribution")
# ... Repeat for Open, High, Low

# Heatmaps
fig_dataX_heatmap = go.Figure(data=go.Heatmap(z=dataX.corr(), x=dataX.columns, y=dataX.columns, colorscale="Reds"))
fig_dataX_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data (During COVID)")

fig_dataY_heatmap = go.Figure(data=go.Heatmap(z=dataY.corr(), x=dataY.columns, y=dataY.columns, colorscale="Blues"))
fig_dataY_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data (Before COVID)")


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
                go.Candlestick(
                    x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='OHLC'
                ),
                go.Scatter(
                    x=data['Date'],
                    y=data['Volume'],
                    name='Volume',
                    yaxis='y2'

                )],
            'layout': {
                'title': 'Open, High, Low, Close over Time',
                'yaxis': {'title': 'Price'},
                'yaxis2': {'title': 'Volume', 'overlaying': 'y', 'side': 'right'},
            }
        }
    ),
    dcc.Graph(figure=fig_dataX_close_box),
    dcc.Graph(figure=fig_dataX_close_hist),
    # ... Add other figures for Open, High, Low
    dcc.Graph(figure=fig_dataY_close_box),
    dcc.Graph(figure=fig_dataY_close_hist),
    # ... Add other figures for Open, High, Low
    dcc.Graph(figure=fig_dataX_heatmap),
    dcc.Graph(figure=fig_dataY_heatmap)
])

if __name__ == '__main__':
    app.run_server(debug=True)