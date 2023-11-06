import numpy as np # linear algebra
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from time import time
import pydot
import graphviz
from keras.models import load_model
import math
import seaborn as sns
import datetime as dt
from datetime import datetime    
sns.set_style("whitegrid")
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from keras.layers import MaxPooling1D, Flatten
from keras.regularizers import L1, L2
from keras.metrics import Accuracy
from keras.metrics import RootMeanSquaredError
from keras.utils import plot_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import explained_variance_score, mean_poisson_deviance, mean_gamma_deviance
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

def generate_signals(self):
    self.calculate_moving_averages()
    signals = pd.DataFrame(index=self.data.index)
    signals['signal'] = 0.0

    # Create signals
    signals['signal'][self.short_window:] = np.where(self.data['short_mavg'][self.short_window:] > self.data['long_mavg'][self.short_window:], 1.0, 0.0)   

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()


#%matplotlib inline
plt.style.use("ggplot")

data = pd.read_csv('SPY.csv')
data.head()
data.info()
data.describe()
data.isnull().sum()
data.reset_index(drop=True, inplace=True)
#data.fillna(data.mean(), inplace=True)
#data.head()

data.plot(legend=True,subplots=True, figsize = (12, 6))
plt.show()


#data['Close'].plot(legend=True, figsize = (12, 6))
#plt.show()
#data['Volume'].plot(legend=True,figsize=(12,7))
#plt.show()

data.shape
data.size
data.describe(include='all').T
data.dtypes
data.nunique()
ma_day = [10,50,100]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    data[column_name]=pd.DataFrame.rolling(data['Close'],ma).mean()

data['Daily Return'] = data['Close'].pct_change()
# plot the daily return percentage
data['Daily Return'].plot(figsize=(12,5),legend=True,linestyle=':',marker='o')
plt.show()

sns.displot(data['Daily Return'].dropna(),bins=100,color='green')
plt.show()

date=pd.DataFrame(data['Date'])
closing_df1 = pd.DataFrame(data['Close'])
close1  = closing_df1.rename(columns={"Close": "data_close"})
close2=pd.concat([date,close1],axis=1)
close2.head()

data.reset_index(drop=True, inplace=True)
#data.fillna(data.mean(), inplace=True)
data.head()

data.nunique()

data.sort_index(axis=1,ascending=True)

cols_plot = ['Open', 'High', 'Low','Close','Volume','MA for 10 days','MA for 50 days','MA for 100 days','Daily Return']
axes = data[cols_plot].plot(marker='.', alpha=0.7, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)

data.isnull().sum()
cols_plot = ['Open', 'High', 'Low','Close']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)

df.describe().transpose()

X = []
Y = []
window_size=100
for i in range(1 , len(df) - window_size -1 , 1):
    first = df.iloc[i,2]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((df.iloc[i + j, 2] - first) / first)
    temp2.append((df.iloc[i + window_size, 2] - first) / first)
    X.append(np.array(temp).reshape(100, 1))
    Y.append(np.array(temp2).reshape(1, 1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_Y = np.array(y_train)
test_Y = np.array(y_test)

train_X = train_X.reshape(train_X.shape[0],1,100,1)
test_X = test_X.reshape(test_X.shape[0],1,100,1)

print(len(train_X))
print(len(test_X))

model = tf.keras.Sequential()

# Creating the Neural Network model here...
# CNN layers
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 100, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
# model.add(Dense(5, kernel_regularizer=L2(0.01)))

# LSTM layers
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dropout(0.5))

#Final layers
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=40,batch_size=40, verbose=1, shuffle =True)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

plt.plot(history.history['mse'], label='train mse')
plt.plot(history.history['val_mse'], label='val mse')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.evaluate(test_X, test_Y)

# predict probabilities for test set
yhat_probs = model.predict(test_X, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]

var = explained_variance_score(test_Y.reshape(-1,1), yhat_probs)
print('Variance: %f' % var)

r2 = r2_score(test_Y.reshape(-1,1), yhat_probs)
print('R2 Score: %f' % var)

var2 = max_error(test_Y.reshape(-1,1), yhat_probs)
print('Max Error: %f' % var2)

predicted  = model.predict(test_X)
test_label = test_Y.reshape(-1,1)
predicted = np.array(predicted[:,0]).reshape(-1,1)
len_t = len(train_X)
for j in range(len_t , len_t + len(test_X)):
    temp = data.iloc[j,3]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted[j - len_t] = predicted[j - len_t] * temp + temp
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.plot(test_label, color = 'red', label = 'Real Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()

#Testing phase
dataX = pd.read_csv('SPY.csv')
dataY = pd.read_csv('SPY.csv')
dataX.info()
dataX.head()
dataX['Date'] = pd.to_datetime(dataX['Date'])
dataY['Date'] = pd.to_datetime(dataY['Date'])

total_data_len = len(data)
testing_data_len = int(0.25 * total_data_len)

testing_start_index = total_data_len - testing_data_len
test_stock_data = data.iloc[testing_start_index:]
testing_start_date = test_stock_data['Date'].iloc[0]
testing_end_date = test_stock_data['Date'].iloc[-1]

# for heatmap
start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2021-11-29')
start = pd.to_datetime('2018-01-01')
end = pd.to_datetime('2020-01-01')

fill = (dataX['Date']>=start_date) & (dataX['Date']<=end_date)
dataX = dataX.loc[fill]
dataX
fill2 = (dataY['Date']>=start) & (dataY['Date']<=end)
dataY = dataY.loc[fill2]
dataY
dataX.describe()
dataY.describe()
sns_plot = sns.histplot(dataX['Close'])
sns_plot2 = sns.histplot(dataY['Close'])
fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= dataX['Close'], ax = ax[0,0])
sns.histplot(dataX['Close'], ax = ax[0,1])
sns.boxplot(x= dataX['Open'], ax = ax[1,0])
sns.histplot(dataX['Open'], ax = ax[1,1])
sns.boxplot(x= dataX['High'], ax = ax[2,0])
sns.histplot(dataX['High'], ax = ax[2,1])
sns.boxplot(x= dataX['Low'], ax = ax[3,0])
sns.histplot(dataX['Low'], ax = ax[3,1])
plt.tight_layout()

fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= dataY["Close"], ax = ax[0,0])
sns.histplot(dataY['Close'], ax = ax[0,1])
sns.boxplot(x= dataY["Open"], ax = ax[1,0])
sns.histplot(dataY['Open'], ax = ax[1,1])
sns.boxplot(x= dataY["High"], ax = ax[2,0])
sns.histplot(dataY['High'], ax = ax[2,1])
sns.boxplot(x= dataY["Low"], ax = ax[3,0])
sns.histplot(dataY['Low'], ax = ax[3,1])
plt.tight_layout()

#heatmaps
plt.figure(figsize=(10,6))
sns.heatmap(dataX.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (During COVID)',
         fontsize=13)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(dataY.corr(),cmap=plt.cm.Blues,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (Before COVID)',
         fontsize=13)
plt.show()

dataX = pd.read_csv('SPY.csv')
dataY = pd.read_csv('SPY.csv')
dataX.info()

dataX['Date'] = pd.to_datetime(dataX['Date'])
dataY['Date'] = pd.to_datetime(dataY['Date'])

start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2021-11-29')
start = pd.to_datetime('2018-01-01')
end = pd.to_datetime('2020-01-01')

fill = (dataX['Date'] >= start_date) & (dataX['Date'] <= end_date)
dataX = dataX.loc[fill]
fill2 = (dataY['Date'] >= start) & (dataY['Date'] <= end)
dataY = dataY.loc[fill2]
fill = (dataX['Date']>=start_date) & (dataX['Date']<=end_date)
dataX = dataX.loc[fill]
dataX
fill2 = (dataY['Date']>=start) & (dataY['Date']<=end)
dataY = dataY.loc[fill2]
dataY
dataX.describe()
dataY.describe()
sns_plot = sns.distplot(dataX['Close'])
sns_plot2 = sns.distplot(dataY['Close'])

fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= dataX["Close"], ax = ax[0,0])
sns.distplot(dataX['Close'], ax = ax[0,1])
sns.boxplot(x= dataX["Open"], ax = ax[1,0])
sns.distplot(dataX['Open'], ax = ax[1,1])
sns.boxplot(x= dataX["High"], ax = ax[2,0])
sns.distplot(dataX['High'], ax = ax[2,1])
sns.boxplot(x= dataX["Low"], ax = ax[3,0])
sns.distplot(dataX['Low'], ax = ax[3,1])
plt.tight_layout()

fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= dataY["Close"], ax = ax[0,0])
sns.distplot(dataY['Close'], ax = ax[0,1])
sns.boxplot(x= dataY["Open"], ax = ax[1,0])
sns.distplot(dataY['Open'], ax = ax[1,1])
sns.boxplot(x= dataY["High"], ax = ax[2,0])
sns.distplot(dataY['High'], ax = ax[2,1])
sns.boxplot(x= dataY["Low"], ax = ax[3,0])
sns.distplot(dataY['Low'], ax = ax[3,1])
plt.tight_layout()

plt.figure(figsize=(10,6))
sns.heatmap(dataX.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (During COVID)',
         fontsize=13)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(dataY.corr(),cmap=plt.cm.Blues,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (Before COVID)',
         fontsize=13)
plt.show()