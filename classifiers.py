import mysql.connector
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Database configuration
config = {
    'user': 'root',
    'password': 'stable',
    'host': '127.0.0.1',
    'database': 'stocks',
    'raise_on_warnings': True
}

def get_all_tickers():
    cnx = mysql.connector.connect(**config)
    query = "SELECT DISTINCT ticker FROM stock_data"
    tickers = pd.read_sql(query, cnx)
    cnx.close()
    return tickers['ticker'].tolist()

# Function to insert classifier results into the database
def insert_classifier_results(ticker, date, svm_accuracy, knn_accuracy):
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    query = """INSERT INTO classifiers (ticker, most_recent_date, svm_accuracy, knn_accuracy)
               VALUES (%s, %s, %s, %s)"""
    cursor.execute(query, (ticker, date, svm_accuracy, knn_accuracy))
    cnx.commit()
    cursor.close()
    cnx.close()

tickers = get_all_tickers()
for ticker in tickers:
    # Fetch data from database
    cnx = mysql.connector.connect(**config)
    query = f"SELECT * FROM stock_data WHERE ticker='{ticker}' ORDER BY date"
    df = pd.read_sql(query, cnx)
    cnx.close()

    # ... [Rest of the processing and training code remains same] ...
    for i in range(1, 6):  # Using past 5 days prices
        df[f'lag_{i}'] = df['close_price'].shift(i)

# Target: 1 if price went up, 0 if it went down or remained the same
    df['target'] = (df['close_price'] > df['close_price'].shift(1)).astype(int)

# Drop missing values (first 5 rows will have NaN because of lags)
    df.dropna(inplace=True)

# Split data
    X = df[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the data (important for SVM and KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# SVM
    clf_svm = SVC()
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# KNN
    clf_knn = KNeighborsClassifier(n_neighbors=5)
    clf_knn.fit(X_train, y_train)
    y_pred_knn = clf_knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

    # Storing classifier results
    most_recent_date = df['date'].max()
    insert_classifier_results(ticker, most_recent_date, accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_knn))



# Feature engineering: Using lagged prices as features
#for i in range(1, 6):  # Using past 5 days prices
#    df[f'lag_{i}'] = df['close_price'].shift(i)

# Target: 1 if price went up, 0 if it went down or remained the same
#df['target'] = (df['close_price'] > df['close_price'].shift(1)).astype(int)

# Drop missing values (first 5 rows will have NaN because of lags)
#df.dropna(inplace=True)

# Split data
#X = df[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]
#y = df['target']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the data (important for SVM and KNN)
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# SVM
#clf_svm = SVC()
#clf_svm.fit(X_train, y_train)
#y_pred_svm = clf_svm.predict(X_test)
#print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# KNN
#clf_knn = KNeighborsClassifier(n_neighbors=5)
#clf_knn.fit(X_train, y_train)
#y_pred_knn = clf_knn.predict(X_test)
#print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
