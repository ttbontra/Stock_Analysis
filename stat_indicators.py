import pandas as pd
import numpy as np
import mysql.connector
from ta import add_all_ta_features
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_DATABASE'),
    'raise_on_warnings': True
}

def fetch_all_tickers():
    cnx = mysql.connector.connect(**config)
    query = "SELECT DISTINCT ticker FROM stock_data"
    cursor = cnx.cursor()
    cursor.execute(query)
    tickers = [row[0] for row in cursor.fetchall()]
    cnx.close()
    return tickers

def fetch_stock_data(ticker):
    cnx = mysql.connector.connect(**config)
    query = f"SELECT date, open_price, high, low, close_price, volume FROM stock_data WHERE ticker='{ticker}' ORDER BY date"
    df = pd.read_sql(query, cnx)
    cnx.close()
    return df

def compute_indicators(df):
    df = add_all_ta_features(df, open="open_price", high="high", low="low", close="close_price", volume="volume")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['momentum_rsi', 'trend_macd'], inplace=True)
    df.fillna(0, inplace=True)
    return df

def insert_indicators_to_db(ticker, df):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    for index, row in df.iterrows():
        date = row['date']

        # Extract indicators
        rsi = row['momentum_rsi']
        macd = round(row['trend_macd'], 4)
        bollinger_upper = round(row['volatility_bbm'], 4)
        bollinger_lower = round(row['volatility_bbl'], 4)
        ppo = round(row['momentum_ppo'], 4)
        stochastic_oscillator = round(row['momentum_stoch'], 4)
        roc = round(row['momentum_roc'], 4)
        #z_score = round(row['volatility_kcc'], 4)

        insert_query = """
            INSERT INTO stock_indicators (ticker, date, rsi, macd, bollinger_upper, bollinger_lower, ppo, stochastic_oscillator, roc)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            rsi=%s, 
            macd=%s, 
            bollinger_upper=%s, 
            bollinger_lower=%s, 
            ppo=%s, 
            stochastic_oscillator=%s, 
            roc=%s
            
        """
        data_to_insert = (ticker, date, rsi, macd, bollinger_upper, bollinger_lower, ppo, stochastic_oscillator, roc, rsi, macd, bollinger_upper, bollinger_lower, ppo, stochastic_oscillator, roc)

        try:
            cursor.execute(insert_query, data_to_insert)
        except mysql.connector.Error as err:
            print(f"Error inserting row for {ticker} on date {date}: {err}")
            #print(f"Query: {insert_query}")
            #rint(f"Data: {data_to_insert}")

    cnx.commit()
    cursor.close()
    cnx.close()

if __name__ == '__main__':
    tickers = fetch_all_tickers()
    
    for ticker in tickers:
        try:
            print(f"Processing {ticker}")
            df = fetch_stock_data(ticker)
            df_indicators = compute_indicators(df)
            insert_indicators_to_db(ticker, df_indicators)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue