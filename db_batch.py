import pandas as pd
import yfinance as yf
import mysql.connector
import schedule
import time

# Database configuration
config = {
    'user': 'root',
    'password': 'stable',
    'host': '127.0.0.1',
    'database': 'stocks',
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
        return None

def save_to_database(df):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    for _, row in df.iterrows():
        date_str = row.get('Date', None)
        if date_str:
            date = date_str.split(' ')[0]  # Extract just the date portion
        else:
            print(f"Missing date for row: {row}")
            continue
        open_price = row.get('Open', None)
        high = row.get('High', None)
        low = row.get('Low', None)
        close_price = row.get('Close', None)
        volume = row.get('Volume', None)
        ticker = row.get('ticker', None)

        insert_query = ("""
            INSERT INTO stock_data (ticker, date, open_price, high, low, close_price, volume) 
            VALUES (%s, %s, %s, %s, %s, %s, %s) 
            ON DUPLICATE KEY UPDATE 
            open_price=VALUES(open_price), 
            high=VALUES(high), 
            low=VALUES(low), 
            close_price=VALUES(close_price), 
            volume=VALUES(volume)
        """)

        try:
            cursor.execute(insert_query, (ticker, date, open_price, high, low, close_price, volume))
        except mysql.connector.DatabaseError as e:
            print(f"Error inserting row: {row}")
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
schedule.every().day.at("14:20").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)



