import pandas as pd
import mysql.connector
from ta import add_all_ta_features
from ta.utils import dropna

def fetch_stock_data(ticker):
    cnx = mysql.connector.connect(**config)
    query = f"SELECT date, open_price, close_price, high, low, volume FROM stock_data_optimized WHERE ticker='{ticker}' ORDER BY date"
    df = pd.read_sql(query, cnx)
    cnx.close()
    return df

def compute_indicators(df):
    # Drop NaN values
    df = dropna(df)
    
    # Compute indicators
    df = add_all_ta_features(df, open="open_price", high="high", low="low", close="close_price", volume="volume")
    
    return df

def save_indicators_to_db(ticker, df):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    for index, row in df.iterrows():
        insert_query = """
        INSERT INTO stock_indicators (ticker, date, rsi, macd, bollinger_upper, bollinger_lower, ppo, stochastic_oscillator, roc, z_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (ticker, row['date'], row['momentum_rsi'], row['trend_macd'], row['volatility_bbm'], row['volatility_bbl'], row['momentum_ppo'], row['momentum_stoch'], row['momentum_roc'], row['close_price_zscore']))
        
    cnx.commit()
    cursor.close()
    cnx.close()

# Example of fetching data for a ticker, computing indicators, and saving them
ticker = "AAPL"
df = fetch_stock_data(ticker)
df = compute_indicators(df)
save_indicators_to_db(ticker, df)
