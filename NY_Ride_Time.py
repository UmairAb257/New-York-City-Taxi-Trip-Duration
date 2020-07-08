import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 16,10
%matplotlib inline
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb

print(os.listdir("C:/Users/Umair/Desktop/ml/Taxi_ride_TimeNY"))

df_test = pd.read_csv("C:/Users/Umair/Desktop/ml/Taxi_ride_TimeNY/test.csv")
df_train = pd.read_csv("C:/Users/Umair/Desktop/ml/Taxi_ride_TimeNY/train.csv")

df_train.head(3)
df_train.columns

# Preprocessing
# how long is the average trip
# '+ 1' to make sure it does not over shot the graph
df_train['log_trip_duration'] = np.log(df_train['trip_duration']).values + 1
plt.hist(df_train['log_trip_duration'].values, bins = 100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of training records')

# Ceck how much overlap is between training and test data. too much overlap results in overfit
N = 10000
city_long_border = (-7, -74)
city_lat_border = (40, 40)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(df_train['pickup_longitude'].values[:N],
              df_train['pickup_latitude'].values[:N],
              color= 'blue', s=1, label='train', alpha=0.1)
ax[1].scatter(df_test['pickup_longitude'].values[:N],
              df_test['pickup_latitude'].values[:N],
              color= 'blue', s=1, label='test', alpha=0.1)

plt.show()

# train model
feature_names = list(df_train.columns)
feature_names.remove('trip_duration')
feature_names
Y = np.log(df_train['trip_duration'].values + 1)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_train[feature_names].values, Y, test_size=0.2, random_state=1987)

model = xgb.train('default', df_train)















