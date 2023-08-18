import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def fetch_sp500_data():
    # Fetch the S&P 500 tickers from Wikipedia
    sp500 = pd.read_html(r'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

    data_dict = {}  # Dictionary to store data for each ticker

    for ticker in sp500:
        try:
            data_dict[ticker] = fetch_data(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    return data_dict

def fetch_data(ticker_symbol, timeframe='1y'):
    """
    Fetches data for the given ticker_symbol and timeframe.
    
    Args:
    - ticker_symbol (str): The stock ticker symbol.
    - timeframe (str): The timeframe for which data is to be fetched. Default is '1y' (1 year).
    
    Returns:
    - pd.DataFrame: A DataFrame containing the stock data.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=timeframe)

        # Drop the 'Dividends' and 'Stock Splits' columns if they exist
        df = df.drop(columns=[col for col in ['Dividends', 'Stock Splits'] if col in df.columns])

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None

def generate_sp500_heatmap():
    data_dict = fetch_sp500_data()
    #df = fetch_sp500_data()
    close_prices = pd.DataFrame({ticker: data['Close'] for ticker, data in data_dict.items() if data is not None})
    corr_matrix = close_prices.corr()
    fig_sp500_heatmap = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale="Greens"))
    fig_sp500_heatmap.update_layout(title="Heatmap displaying the relationship between the features of the data")
    return fig_sp500_heatmap

def generate_sp500_treemap():
    df = fetch_sp500_data()
    color_bin = [-1, -0.02, -0.01, 0, 0.01, 0.02, 1]
    df['colors'] = pd.cut(df['delta'], bins=color_bin, labels=['red', 'indianred', 'lightpink', 'lightgreen', 'lime', 'green'])

    fig = px.treemap(df, path=['sector', 'ticker'], values='market_cap', color='colors',
                     color_discrete_map={'(?)': '#262931', 'red': 'red', 'indianred': 'indianred', 'lightpink': 'lightpink', 'lightgreen': 'lightgreen', 'lime': 'lime', 'green': 'green'},
                     hover_data={'delta': ':.2p'})
    return fig

fig1 = generate_sp500_heatmap()
fig1.show()
fig2 = generate_sp500_treemap()
fig2.show()