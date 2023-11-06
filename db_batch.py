import pandas as pd
import yfinance as yf
import mysql.connector
import schedule
import time
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

def fetch_data(ticker_symbol, timeframe='1y'):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=timeframe)
        df = df.drop(columns=[col for col in ['Dividends', 'Stock Splits'] if col in df.columns])
        # Add ticker symbol as a new column
        df['ticker'] = ticker_symbol
        return df

    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        print(df)
        return None

def save_to_database(df):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    for index, row in df.iterrows():
        date_str = str(index)
        date = date_str.split(' ')[0]  # Extract just the date portion
        
        # Ensure the data types are as expected
        try:
            open_price = round(float(row['Open']), 2)
            high = round(float(row['High']), 2)
            low = round(float(row['Low']), 2)
            close_price = round(float(row['Close']), 2)
            volume = int(row.get('Volume', None))
            ticker = str(row['ticker'])
        except Exception as e:
            print(f"Error casting data: {e}")
            print(row)
            continue
        
        ticker = row.get('ticker', None)

        insert_query = ("""
            INSERT INTO stock_data (ticker, date, open_price, high, low, close_price, volume) 
            VALUES (%s, %s, %s, %s, %s, %s, %s) 
            ON DUPLICATE KEY UPDATE 
            open_price=%s, 
            high=%s, 
            low=%s, 
            close_price=%s, 
            volume=%s
        """)

        try:
            cursor.execute(insert_query, (ticker, date, open_price, high, low, close_price, volume, open_price, high, low, close_price, volume))
        except mysql.connector.DatabaseError as e:
            print(f"Error inserting row: {row}")
            print(f"Values: Ticker={ticker}, Date={date}, Open={open_price}, High={high}, Low={low}, Close={close_price}, Volume={volume}")
            print(f"Error message: {e}")

    cnx.commit()
    cursor.close()
    cnx.close()

def job():
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f]

    for ticker in tickers:
        df = fetch_data(ticker)
        if df is not None:
            save_to_database(df)
        time.sleep(5)  # Avoid hitting rate limits

# Schedule the job to run at the end of every day (e.g., 11:59 PM)
schedule.every().day.at("16:33").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)



