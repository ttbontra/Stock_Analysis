import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf  
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

def generate_candlestick(data):
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
    return fig_candlestick

def generate_daily_return(data):
    fig_daily_return = px.line(data, x=data.index, y='Daily Return', title='Daily Return', markers=True, line_shape='linear')
    return fig_daily_return

def generate_histogram(data):
    fig_histogram = px.histogram(data, x='Daily Return', nbins=100, color_discrete_sequence=['green'])
    return fig_histogram

def generate_box_plots(data):
    fig_box_plots = make_subplots(rows=1, cols=4, shared_xaxes=True, vertical_spacing=0.02)
    fig_box_plots.add_trace(go.Box(y=data["Open"], name="Open Box Plot", boxmean=True), row=1, col=1)
    fig_box_plots.add_trace(go.Box(y=data["High"], name="High Box Plot", boxmean=True), row=1, col=2)
    fig_box_plots.add_trace(go.Box(y=data["Low"], name="Low Box Plot", boxmean=True), row=1, col=3)
    fig_box_plots.add_trace(go.Box(y=data["Close"], name="Close Box Plot", boxmean=True), row=1, col=4)
    fig_box_plots.update_layout(height=500, title_text="Box Plots")
    return fig_box_plots

def generate_2box_plots(data):
    fig_box_plots = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.02)
    fig_box_plots.add_trace(go.Box(y=data["Low"], name="Low Box Plot", boxmean=True), row=1, col=1)
    fig_box_plots.add_trace(go.Box(y=data["Close"], name="Close Box Plot", boxmean=True), row=1, col=2)
    fig_box_plots.update_layout(height=500, title_text="Box Plots")
    return fig_box_plots

def generate_combined_distribution(data):
    # Create the subplots layout
    fig_combined_hist = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.02)

    # Close Distribution
    close_hist = px.histogram(data, x="Close", title="Close Distribution")
    for trace in close_hist.data:
        fig_combined_hist.add_trace(trace, row=1, col=1)

    # Open Distribution
    open_hist = px.histogram(data, x="Open", title="Open Distribution")
    for trace in open_hist.data:
        fig_combined_hist.add_trace(trace, row=1, col=2)

    # Update layout
    fig_combined_hist.update_layout(height=500, title_text="Close and Open Distributions")

    return fig_combined_hist

def generate_close_distribution(data):
    fig_dataX_close_hist = px.histogram(data, x="Close", title="Close Distribution")
    return fig_dataX_close_hist

def generate_open_distribution(data):
    fig_dataY_close_hist = px.histogram(data, x="Open", title="Open Distribution")
    return fig_dataY_close_hist

def generate_heatmap_during_covid(data):
    fig_dataX_heatmap = go.Figure(data=go.Heatmap(z=data.corr(), x=data.columns, y=data.columns, colorscale="Reds"))
    fig_dataX_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data (During COVID)")
    return fig_dataX_heatmap

def generate_heatmap_before_covid(data):
    fig_dataY_heatmap = go.Figure(data=go.Heatmap(z=data.corr(), x=data.columns, y=data.columns, colorscale="Blues"))
    fig_dataY_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data (Before COVID)")
    return fig_dataY_heatmap

config = {
    'user': 'root',
    'password': 'stable',
    'host': '127.0.0.1',
    'database': 'stocks',
    'raise_on_warnings': True
}

# Connect to the database and fetch data
cnx = mysql.connector.connect(**config)
query = "SELECT * FROM classifiers"  # Adjust if the table name is different
df = pd.read_sql(query, cnx)
cnx.close()

# 1. Heatmap:
# As previously mentioned, heatmaps work best with matrix-like data, 
# so we need a bit more data processing if we're to use it meaningfully here.
# A basic heatmap example with SVM and KNN accuracies is as follows:
correlation_matrix = df[['svm_accuracy', 'knn_accuracy']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Between SVM and KNN Accuracies')
plt.show()

# 2. Histograms:
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['svm_accuracy'], kde=True, bins=30, color='blue')
plt.title('Distribution of SVM Accuracy')

plt.subplot(1, 2, 2)
sns.histplot(df['knn_accuracy'], kde=True, bins=30, color='green')
plt.title('Distribution of KNN Accuracy')

plt.tight_layout()
plt.show()

# 3. Scatter Plot:
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df, x='svm_accuracy', y='knn_accuracy', alpha=0.7, color='red')
plt.title('SVM vs. KNN Accuracy for Each Ticker')
plt.xlabel('SVM Accuracy')
plt.ylabel('KNN Accuracy')
plt.grid(True, which="both", ls="--")
plt.show()
