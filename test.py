import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Database configuration
config = {
    'user': 'root',
    'password': 'stable',
    'host': '127.0.0.1',
    'database': 'stocks',
    'raise_on_warnings': True
}

# Connect to the database and fetch data
cnx = mysql.connector.connect(**config)
query = "SELECT * FROM classifiers"  # Adjust if the table name is different
df = pd.read_sql(query, cnx)
cnx.close()

# 1. Heatmap:
# As previously mentioned, heatmaps work best with matrix-like data, 
# so we need a bit more data processing if we're to use it meaningfully here.
# A basic heatmap example with SVM and KNN accuracies is as follows:
correlation_matrix = df[['svm_accuracy', 'knn_accuracy']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Between SVM and KNN Accuracies')
plt.show()

# 2. Histograms:
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['svm_accuracy'], kde=True, bins=30, color='blue')
plt.title('Distribution of SVM Accuracy')

plt.subplot(1, 2, 2)
sns.histplot(df['knn_accuracy'], kde=True, bins=30, color='green')
plt.title('Distribution of KNN Accuracy')

plt.tight_layout()
plt.show()

# 3. Scatter Plot:
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df, x='svm_accuracy', y='knn_accuracy', alpha=0.7, color='red')
plt.title('SVM vs. KNN Accuracy for Each Ticker')
plt.xlabel('SVM Accuracy')
plt.ylabel('KNN Accuracy')
plt.grid(True, which="both", ls="--")
plt.show()
