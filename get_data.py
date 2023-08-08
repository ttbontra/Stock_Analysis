import pandas as pd
import yfinance as yf

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