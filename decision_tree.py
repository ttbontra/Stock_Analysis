import mysql.connector
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Database configuration
config = {
    'user': 'root',
    'password': 'stable',
    'host': '127.0.0.1',
    'database': 'stocks',
    'raise_on_warnings': True
}

# Fetch data from database
cnx = mysql.connector.connect(**config)
query = "SELECT * FROM stock_indicators WHERE ticker='AAPL' ORDER BY date"
df = pd.read_sql(query, cnx)
cnx.close()

# Define features and target
features = ['rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'ppo', 'stochastic_oscillator', 'roc']
X = df[features]
y = (df['close_price'].shift(-1) > df['close_price']).astype(int)  # 1 if price goes up next day, 0 otherwise
y = y[:-1]  # remove last row as we don't have a target for it
X = X[:-1]  # match the shape with y

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=3)  # Limiting the depth for simplicity
clf.fit(X_train, y_train)

# Predict and check accuracy
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot decision tree
plot_tree(clf, feature_names=features, class_names=['Down', 'Up'], filled=True)
