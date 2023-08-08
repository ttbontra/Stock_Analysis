import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from pandas_datareader import data as web
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
import yfinance as yf  
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
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QProgressBar, QLabel, QTabWidget
from PyQt5.QtCore import QUrl, QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebEngineWidgets import QWebEnginePage
from forecast import train_model, forecast
from get_data import fetch_data


# Load and preprocess the data
#data = pd.read_csv('SPY.csv')
#data.dropna(inplace=True)
#data['Daily Return'] = data['Close'].pct_change()
#dataX = data
#dataY = data
#dataX['Date'] = pd.to_datetime(dataX['Date'])
#dataY['Date'] = pd.to_datetime(dataY['Date'])


fig_candlestick = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig_candlestick.add_trace(
    go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ),
    row=1, col=1
)

fig_candlestick.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data['Volume'],
        name='Volume'
    ),
    row=2, col=1
)
fig_candlestick.update_layout(height=500, title_text="Candlestick Chart with Rangeslider")

fig_box_plots = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.02)
fig_box_plots.add_trace(
    go.Box( 
        y=data["Open"],
        name="Open Box Plot",
        boxmean=True, 
        ),
    row=1, col=1
)

fig_box_plots.add_trace(
    go.Box( 
        y=data["High"],
        name="High Box Plot",
        boxmean=True, 
        ),
    row=1, col=2
)
fig_box_plots.update_layout(height=500, title_text="Box Plots")

# Box Plots and Distribution Plots for dataX
fig_dataX_close_box = px.box(data, y="Close", title="Close Box Plot")
fig_dataX_close_hist = px.histogram(data, x="Close", title="Close Distribution")

# Box Plots and Distribution Plots for dataY
fig_dataY_close_box = px.box(data, y="Open", title="Open Box Plot")
fig_dataY_close_hist = px.histogram(data, x="Open", title="Open Distribution")


# Heatmaps
fig_dataX_heatmap = go.Figure(data=go.Heatmap(z=data.corr(), x=data.columns, y=data.columns, colorscale="Reds"))
fig_dataX_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data (During COVID)")

fig_dataY_heatmap = go.Figure(data=go.Heatmap(z=data.corr(), x=data.columns, y=data.columns, colorscale="Blues"))
fig_dataY_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data (Before COVID)")


# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app = dash.Dash(__name__)
# Define the app layout
app.layout = html.Div([
    dcc.Input(id='ticker-input', type='text', placeholder='Enter Ticker Symbol'),
    dcc.Dropdown(
        id='timeframe-dropdown',
        options=[
            {'label': '1 Month', 'value': '1mo'},
            {'label': '3 Months', 'value': '3mo'},
            {'label': '6 Months', 'value': '6mo'},
            {'label': '1 Year', 'value': '12mo'},
            {'label': '5 Years', 'value': '60mo'},
            {'label': '10 Years', 'value': '120mo'},
            {'label': '20 Years', 'value': '240mo'}
        ],
        value='24mo'
    ),
    html.Button('Fetch Data', id='fetch-button'),
    html.Div(id='output-div'),
    html.Button('Train and Forecast', id='train-forecast-button', n_clicks=0),
    # Candlestick plot of high, low, open, close and volume
    dcc.Graph(
        id='ohlc', figure=fig_candlestick
    ),
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

    dcc.Graph(figure=fig_box_plots),
    dcc.Graph(figure=fig_dataX_close_hist),
    dcc.Graph(figure=fig_dataY_close_hist),
    dcc.Graph(figure=fig_dataX_heatmap),
    dcc.Graph(figure=fig_dataY_heatmap)
])

@app.callback(
    [Output('ohlc', 'figure'),
     Output('daily-return', 'figure'),
     Output('histogram', 'figure'),
     Output('box-plots', 'figure'),
     Output('dataX-close-hist', 'figure'),
     Output('dataY-close-hist', 'figure'),
     Output('dataX-heatmap', 'figure'),
     Output('dataY-heatmap', 'figure'),
     Output('output-div', 'children')],
    [Input('fetch-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value'),
     dash.dependencies.State('timeframe-dropdown', 'value')]
)
def update_output(n_clicks, ticker_value, timeframe_value):
    if n_clicks:
        data = fetch_data(ticker_value, timeframe_value)
        if data is not None:
            # Preprocess the data
            data.dropna(inplace=True)
            data['Daily Return'] = data['Close'].pct_change()

            # Create the various plots
            fig_candlestick = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
            fig_candlestick.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )
            fig_candlestick.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume'
                ),
                row=2, col=1
            )
            fig_candlestick.update_layout(height=500, title_text="Candlestick Chart with Rangeslider")

            fig_daily_return = px.line(data, x=data.index, y='Daily Return', title='Daily Return', markers=True, line_shape='linear')
            fig_histogram = px.histogram(data, x='Daily Return', nbins=100, color_discrete_sequence=['green'])

            fig_box_plots = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.02)
            fig_box_plots.add_trace(go.Box(y=data["Open"], name="Open Box Plot", boxmean=True), row=1, col=1)
            fig_box_plots.add_trace(go.Box(y=data["High"], name="High Box Plot", boxmean=True), row=1, col=2)
            fig_box_plots.update_layout(height=500, title_text="Box Plots")

            fig_dataX_close_hist = px.histogram(data, x="Close", title="Close Distribution")
            fig_dataY_close_hist = px.histogram(data, x="Open", title="Open Distribution")

            fig_dataX_heatmap = go.Figure(data=go.Heatmap(z=data.corr(), x=data.columns, y=data.columns, colorscale="Reds"))
            fig_dataX_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data (During COVID)")

            fig_dataY_heatmap = go.Figure(data=go.Heatmap(z=data.corr(), x=data.columns, y=data.columns, colorscale="Blues"))
            fig_dataY_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data (Before COVID)")

            return fig_candlestick, fig_daily_return, fig_histogram, fig_box_plots, fig_dataX_close_hist, fig_dataY_close_hist, fig_dataX_heatmap, fig_dataY_heatmap, f"Data fetched for {ticker_value} for the last {timeframe_value}."

        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Error fetching data for {ticker_value}."

if __name__ == '__main__':
    app.run_server(debug=True)