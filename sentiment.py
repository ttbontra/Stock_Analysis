import plotly.graph_objects as go
from textblob import TextBlob
import pandas as pd
import numpy as np

data = pd.read_csv('TSLA_History.csv')

def analyze_stock_sentiment(data):
    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    data['Volatility'] = data['High'] - data['Low']
    data['NormalizedVolume'] = (data['Volume'] - data['Volume'].min()) / (data['Volume'].max() - data['Volume'].min())
    data['SentimentScore'] = data['Momentum'] - data['Volatility'] + data['NormalizedVolume']
    data['SentimentScore'] = (data['SentimentScore'] - data['SentimentScore'].min()) / (data['SentimentScore'].max() - data['SentimentScore'].min()) * 2 - 1
    return data['SentimentScore'].iloc[-1]

def create_bullet_graph(sentiment_score):
    ranges = [-1, -0.5, 0, 0.5, 1]
    range_colors = ['red', 'orange', 'yellow', 'green']

    # Create bullet graph
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[sentiment_score],
        y=[0.5],  # Center the bar within the range
        orientation='h',
        marker=dict(color='black', line=dict(width=3, color='black')),
        width=0.25  # Adjust the width of the bar to make it thinner
    ))

    for i in range(len(ranges) - 1):
        fig.add_shape(
            type="rect",
            x0=ranges[i],
            x1=ranges[i+1],
            y0=0,
            y1=1,  # Adjust the height of the colored ranges
            fillcolor=range_colors[i],
            line=dict(width=0),
            layer="below"
        )

    fig.update_layout(
        title="Stock Sentiment",
        xaxis=dict(range=[-1, 1], title="Sentiment Score"),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        width=500,
        height=200
    )
    return fig

sentiment_score = analyze_stock_sentiment(data)  # Assuming 'data' is your stock data DataFrame
fig = create_bullet_graph(sentiment_score)
fig.show()
