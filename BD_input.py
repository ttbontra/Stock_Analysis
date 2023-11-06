import csv
import mysql.connector
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

# Connect to the MySQL database
cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()

# List all CSV files with '_history' suffix
csv_files = [f for f in os.listdir() if f.endswith('_history.csv')]

# For each CSV file, read its data and insert into the database
for file in csv_files:
    ticker = file.replace('_history.csv', '')  # Extract ticker from filename

    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            date = row[0].split(' ')[0]  # Extract just the date portion
            open_price = round(float(row[1]), 2)
            high = round(float(row[2]), 2)
            low = round(float(row[3]), 2)
            close_price = round(float(row[4]), 2)
            volume = int(row[5])
            insert_query = ("""
                            INSERT INTO stock_data (ticker, date, open_price, high, low, close_price, volume) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s) 
                            AS new_data
                            ON DUPLICATE KEY UPDATE 
                            open_price=new_data.open_price, 
                            high=new_data.high, 
                            low=new_data.low, 
                            close_price=new_data.close_price, 
                            volume=new_data.volume
                            """)
            try:
                cursor.execute(insert_query, (ticker, date, open_price, high, low, close_price, volume))
            except mysql.connector.DatabaseError as e:
                print(f"Error inserting row: {row}")
                print(f"Error message: {e}")
            #cursor.execute(insert_query, (ticker, date, open_price, high, low, close_price, volume))

# Commit changes and close the connection
cnx.commit()
cursor.close()
cnx.close()
