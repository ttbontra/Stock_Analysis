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
#from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QProgressBar, QLabel, QTabWidget
#from PyQt5.QtCore import QUrl, QThread, pyqtSignal
#from PyQt5.QtWebEngineWidgets import QWebEngineView
#from PyQt5.QtWebEngineWidgets import QWebEnginePage
from forecast import train_and_forecast
from get_data import fetch_data
import sentiment
from sentiment import analyze_stock_sentiment, create_bullet_graph

# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dcc.Input(id='ticker-input', type='text', placeholder='Enter Ticker Symbol', style={'display': 'flex', 'justifyContent': 'center', 'width': '40%', 'padding': '12px', }),
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
        value='250mo',
        style={'width': '40%'}
    ),
    html.Button('Fetch Data', id='fetch-button'),
    html.Div(id='output-div'),
    dcc.Graph(id='ohlc'),
    html.Button('Train and Forecast', id='train-forecast-button', n_clicks=0),
    dcc.Graph(id='forecast-graph'),
    dcc.Graph(id='daily-return'),
    dcc.Graph(id='histogram'),
    dcc.Graph(id='sentiment-bullet-chart'),
    dcc.Graph(id='box-plots'),
    dcc.Graph(id='dataX-close-hist'),
    dcc.Graph(id='dataY-close-hist'),
    dcc.Graph(id='dataX-heatmap'),
    dcc.Graph(id='dataY-heatmap'),

])

@app.callback(
    [Output('ohlc', 'figure'),
     Output('forecast-graph', 'figure'),
     Output('daily-return', 'figure'),
     Output('histogram', 'figure'),
     Output('sentiment-bullet-chart', 'figure'),
     Output('box-plots', 'figure'),
     Output('dataX-close-hist', 'figure'),
     Output('dataY-close-hist', 'figure'),
     Output('dataX-heatmap', 'figure'),
     Output('dataY-heatmap', 'figure'),
     Output('output-div', 'children')],
    [Input('fetch-button', 'n_clicks'),
     Input('train-forecast-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('timeframe-dropdown', 'value')]
)



def combined_callback(n_clicks_fetch, n_clicks_train, ticker_value, timeframe_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [dash.no_update] * 11
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'fetch-button' and ticker_value:
        results = fetch_data_and_update_graphs(ticker_value, timeframe_value)
        if len(results) == 11:  # Ensure fetch_data_and_update_graphs returns 10 items
            return results
        else:
            raise ValueError("fetch_data_and_update_graphs should return a list of 10 items.")
    elif button_id == 'train-forecast-button' and ticker_value:
        forecast_fig, message = handle_train_forecast_button_click(n_clicks_train, ticker_value, timeframe_value)
        return [dash.no_update, forecast_fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, message]
    else:
        return [dash.no_update] * 11
    
def generate_graphs_from_data(data, ticker_value, timeframe_value):
    if data is not None:
            return [dash.no_update] * 11
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
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

    # Return all the figures and the message
    return [fig_candlestick, fig_daily_return, fig_histogram, fig_box_plots, fig_dataX_close_hist, fig_dataY_close_hist, fig_dataX_heatmap, fig_dataY_heatmap, dash.no_update, f"Data fetched for {ticker_value} for the last {timeframe_value}."]


def handle_fetch_button_click(n_clicks, ticker_value, timeframe_value):
    if n_clicks:
        #print(data.head())
        return fetch_data_and_update_graphs(ticker_value, timeframe_value)
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ""

def create_forecast_plot(test_stock_data_processed, predicted_stock_price, tickers):
    x_actual = np.arange(test_stock_data_processed.shape[0])
    x_predicted = np.arange(test_stock_data_processed.shape[0], test_stock_data_processed.shape[0] + predicted_stock_price.shape[0])

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_actual, y=test_stock_data_processed.flatten(), mode='lines', name=f'Actual {tickers} Stock Price'))
    fig.add_trace(go.Scatter(x=x_predicted, y=predicted_stock_price.flatten(), mode='lines', name=f'Predicted {tickers} Stock Price'))
    fig.update_layout(title=f'{tickers} Stock Price Prediction', xaxis_title='Date', yaxis_title=f'{tickers} Stock Price')
    return fig

def handle_train_forecast_button_click(n_clicks, ticker_value, timeframe_value):
    graphs = []
    if n_clicks and ticker_value:
        x_actual, actual_prices, x_predicted, predicted_prices = train_and_forecast(ticker_value)
        print(x_actual)
        print(actual_prices)
        print(x_predicted)
        print(predicted_prices)
        # Create the figure using the returned data
        forecast_fig = create_forecast_plot(actual_prices, predicted_prices, ticker_value)
        #fig = go.Figure()
        #fig.add_trace(go.Scatter(x=x_actual, y=actual_prices, mode='lines', name='Actual Prices'))
        #fig.add_trace(go.Scatter(x=x_predicted, y=predicted_prices, mode='lines', name='Predicted Prices'))
        graphs.append(forecast_fig)
        other_graphs = generate_graphs_from_data(data, ticker_value, timeframe_value)
        graphs.extend(other_graphs)
        print(forecast_fig)
        return forecast_fig, f"Forecast generated for {ticker_value}."
    return dash.no_update, ""

def fetch_data_and_update_graphs(ticker_value, timeframe_value):
    data = fetch_data(ticker_value, timeframe_value)
    #print(data.head())
    if data is not None:
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
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
            sentiment_score = analyze_stock_sentiment(data)
            fig_sentiment_bullet = create_bullet_graph(sentiment_score)
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

            return fig_candlestick, dash.no_update, fig_daily_return, fig_histogram, fig_sentiment_bullet, fig_box_plots, fig_dataX_close_hist, fig_dataY_close_hist, fig_dataX_heatmap, fig_dataY_heatmap, f"Data fetched for {ticker_value} for the last {timeframe_value}."
        
    
    else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Error fetching data for {ticker_value}.", dash.no_update

   
if __name__ == '__main__':
    app.run_server(debug=True)