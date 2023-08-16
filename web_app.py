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
from forecast import train_and_forecast
from get_data import fetch_data
from sentiment import analyze_stock_sentiment, create_bullet_graph
from graphs import (generate_candlestick, generate_daily_return, generate_histogram, 
                    generate_box_plots, generate_close_distribution, generate_open_distribution, 
                    generate_heatmap_during_covid, generate_heatmap_before_covid)

# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dcc.Input(id='ticker-input', type='text', placeholder='Enter Ticker Symbol', className='centered-item'),
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
        style={'width': '50%'}, className='centered-item'),
    html.Button('Fetch Data', id='fetch-button', className='centered-item'),
    html.Div(id='output-div'),
    dcc.Graph(id='ohlc', className='candlestick-chart'),
    html.Button('Train and Forecast', id='train-forecast-button', n_clicks=0),
    dcc.Graph(id='forecast-graph', className='forecast-graph'),
    dcc.Graph(id='daily-return', className='daily-return-chart'),
    html.Div([
        dcc.Graph(id='histogram', className='histogram'),
        dcc.Graph(id='sentiment-bullet-chart', className='bullet-chart'),
    ], className='row-content'),
    #dcc.Graph(id='histogram', style={'flex': '1'}),
    #dcc.Graph(id='sentiment-bullet-chart', style={'flex': '1'}),
    dcc.Graph(id='box-plots'),
    dcc.Graph(id='dataX-close-hist'),
    dcc.Graph(id='dataY-close-hist'),
    dcc.Graph(id='dataX-heatmap'),
    dcc.Graph(id='dataY-heatmap'),

], style={'display': 'flex'}, className='centered-container')

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

    fig_candlestick = generate_candlestick(data)
    fig_daily_return = generate_daily_return(data)
    fig_histogram = generate_histogram(data)
    fig_box_plots = generate_box_plots(data)
    fig_dataX_close_hist = generate_close_distribution(data)
    fig_dataY_close_hist = generate_open_distribution(data)
    fig_dataX_heatmap = generate_heatmap_during_covid(data)
    fig_dataY_heatmap = generate_heatmap_before_covid(data)

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
    if data is not None:
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data.dropna(inplace=True)
        data['Daily Return'] = data['Close'].pct_change()

        fig_candlestick = generate_candlestick(data)
        fig_daily_return = generate_daily_return(data)
        fig_histogram = generate_histogram(data)
        sentiment_score = analyze_stock_sentiment(data)
        fig_sentiment_bullet = create_bullet_graph(sentiment_score)
        fig_box_plots = generate_box_plots(data)
        fig_dataX_close_hist = generate_close_distribution(data)
        fig_dataY_close_hist = generate_open_distribution(data)
        fig_dataX_heatmap = generate_heatmap_during_covid(data)
        fig_dataY_heatmap = generate_heatmap_before_covid(data)

        return fig_candlestick, dash.no_update, fig_daily_return, fig_histogram, fig_sentiment_bullet, fig_box_plots, fig_dataX_close_hist, fig_dataY_close_hist, fig_dataX_heatmap, fig_dataY_heatmap, f"Data fetched for {ticker_value} for the last {timeframe_value}."
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Error fetching data for {ticker_value}.", dash.no_update

   
if __name__ == '__main__':
    app.run_server(debug=True)