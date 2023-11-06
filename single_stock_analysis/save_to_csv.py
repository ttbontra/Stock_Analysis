import pandas as pd
import yfinance

# Load the CSV file with the list of tickers
tickers_df = pd.read_csv('tickers.csv')

# Iterate over the tickers
for ticker_symbol in tickers_df['tickers']:  # assuming 'ticker' is the column name containing the tickers
    try:
        ticker = yfinance.Ticker(ticker_symbol)
        df = ticker.history(period='250mo')

        # Drop the 'Dividends' and 'Stock Splits' columns
        df = df.drop(columns=['Dividends', 'Stock Splits'])

        # Save to a CSV file
        df.reset_index().to_csv(f'etfs/{ticker_symbol}_history.csv', index=False)

        print(f"Data fetched for {ticker_symbol}")
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")

#ticker_symbol = 'AAPL'
#ticker = yfinance.Ticker(ticker_symbol)
#df = ticker.history(period = '250mo') #36 works can try more
# Drop the 'Dividends' and 'Capital Gains' columns
#df = df.drop(columns=['Dividends', 'Stock Splits']) #, 'Capital Gains' can be added back in later

# Save to a CSV file
#df.reset_index().to_csv(f'{ticker_symbol}_history.csv', index=False)

print(df)