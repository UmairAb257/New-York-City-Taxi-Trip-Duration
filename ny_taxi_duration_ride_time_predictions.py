# -*- coding: utf-8 -*-
"""NY Taxi Duration Ride Time Predictions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yHpsw10hxRzNUzQjQePKDE2rLDd800m2
"""

# Commented out IPython magic to ensure Python compatibility.
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 16,10
# %matplotlib inline
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import plot_tree


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# samplesubmit - 1nKhBVTOvnFv4SsrAe4nghT5qh6rMRb1n
# test - 1iNR8EjaffbpaGzGpfVfruPtXE9rEWc8t
# Fastest route test- 1u8AImxQkeWP4RK2ac1f7pNcw0NxBTc9s
# train - 1J9G-EtzGdIFEFdeu-j2sjV5WOPlZ6o-G
# fastest route train 1 - 1Sz319v12PCD0_klk-S3oeUUI3wGfNBXp
# fastest route train 2 - 1lfev5OB2vSqoaz7itwo2KBDERuAN9C3H
# weather data NY Central Park 2016 - 1-i_-hOfrbbF3rO-K05fGiURTzYd3hEg0

sample_submission_downloaded = drive.CreateFile({'id': '1nKhBVTOvnFv4SsrAe4nghT5qh6rMRb1n'})
sample_submission_downloaded.GetContentFile('sample_submission.csv')

train_downloaded = drive.CreateFile({'id': '1J9G-EtzGdIFEFdeu-j2sjV5WOPlZ6o-G'})
train_downloaded.GetContentFile('train.csv')
test_downloaded = drive.CreateFile({'id': '1iNR8EjaffbpaGzGpfVfruPtXE9rEWc8t'})
test_downloaded.GetContentFile('test.csv')

fastest_routes_test_downloaded = drive.CreateFile({'id': '1u8AImxQkeWP4RK2ac1f7pNcw0NxBTc9s'})
fastest_routes_test_downloaded.GetContentFile('fastest_routes_test.csv')
fastest_routes_train_1_downloaded = drive.CreateFile({'id': '1Sz319v12PCD0_klk-S3oeUUI3wGfNBXp'})
fastest_routes_train_1_downloaded.GetContentFile('fastest_routes_train_1.csv')
fastest_routes_train_2_downloaded = drive.CreateFile({'id': '1lfev5OB2vSqoaz7itwo2KBDERuAN9C3H'})
fastest_routes_train_2_downloaded.GetContentFile('fastest_routes_train_2.csv')
weather_data_downloaded = drive.CreateFile({'id': '1-i_-hOfrbbF3rO-K05fGiURTzYd3hEg0'})
weather_data_downloaded.GetContentFile('weather_data_nyc_centralpark_2016.csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
fastest_routes_test = pd.read_csv('fastest_routes_test.csv')
fastest_routes_train_1 = pd.read_csv('fastest_routes_train_1.csv')
fastest_routes_train_2 = pd.read_csv('fastest_routes_train_2.csv')
weather_data = pd.read_csv('weather_data_nyc_centralpark_2016.csv')


#train.info()
################################################################################################################

train.passenger_count = train.passenger_count.astype(np.uint8)
train.vendor_id = train.vendor_id.astype(np.uint8)
train.trip_duration = train.trip_duration.astype(np.uint32)
for c in [c for c in train.columns if c.endswith('tude')]:
    train.loc[:,c] = train[c].astype(np.float32)
print(train.memory_usage().sum()/2**20)

############################################################################################################
train.pickup_datetime=pd.to_datetime(train.pickup_datetime)
train.dropoff_datetime=pd.to_datetime(train.dropoff_datetime)
train['pu_hour'] = train.pickup_datetime.dt.hour
train['yday'] = train.pickup_datetime.dt.dayofyear
train['wday'] = train.pickup_datetime.dt.dayofweek
train['month'] = train.pickup_datetime.dt.month

################################################################################################################
#sns.set_style('white')
#sns.set_context("paper",font_scale=2)
#corr = train.corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#f, ax = plt.subplots(figsize=(11,9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
#           square=True, linewidths=0.5, cbar_kws={"shrink":0.5});

################################################################################################################

# fig, ax = plt.subplots(ncols=1, nrows=1)
# sns.distplot(df['trip_duration']/3600,ax=ax,bins=100,kde=False,hist_kws={'log':True})

#fig, ax = plt.subplots(ncols=1, nrows=1)
#ax.set_xlim(0,30)
#sns.distplot(train['trip_duration']/3600,ax=ax,bins=1000,kde=False,hist_kws={'log':True});


################################################################################################################
def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    miles = km *  0.621371
    return miles

### Distance
# This uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is, the shortest distance over the earth’s surface 
#– giving an ‘as-the-crow-flies’ distance between the points (ignoring any hills they fly over, of course!).
# 
# Haversine formula:	
#           a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
#           c = 2 ⋅ atan2( √a, √(1−a) )
#           d = R ⋅ c
#      where	φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
#           note that angles need to be in radians to pass to trig functions!

train['distance'] = haversine(train.pickup_longitude, train.pickup_latitude, train.dropoff_longitude, train.dropoff_latitude)

weather_data['date']=pd.to_datetime(weather_data.date,format='%d-%m-%Y')
weather_data['yday'] = weather_data.date.dt.dayofyear
#weather_data.head()

weather_data['snow fall'] = weather_data['snow fall'].replace(['T'],0.05).astype(np.float32)
weather_data['precipitation'] = weather_data['precipitation'].replace(['T'],0.05).astype(np.float32)
weather_data['snow depth'] = weather_data['snow depth'].replace(['T'],0.05).astype(np.float32)
#list(weather_data.columns)

train = pd.merge(train,weather_data,on='yday')

train = train.drop(['date','maximum temperature','minimum temperature'],axis=1)
#list(train.columns)


################################################################################################################
#sns.set_style('white')
#sns.set_context("paper",font_scale=2)
#corr = train.corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#f, ax = plt.subplots(figsize=(11,9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
#           square=True, linewidths=0.5, cbar_kws={"shrink":0.5})


# corr
################################################################################################################
fastest_routes_train = pd.concat([fastest_routes_train_1,fastest_routes_train_2],ignore_index=True)
#list(fastest_routes_train.columns)

fastest_routes_train = fastest_routes_train.drop(['step_location_list','step_direction','step_maneuvers','travel_time_per_step','distance_per_step','street_for_each_step','number_of_steps','starting_street','end_street'],axis=1)
#fastest_routes_train.head()

train = pd.merge(train,fastest_routes_train,on='id',how='outer')
#train.head()

mask = ((train.trip_duration > 60) & (train.distance < 0.05))
train = train[~mask]
mask = (train.trip_duration < 60) 
train = train[~mask]
mask =  train.trip_duration > 79200
train = train[~mask]
mask = train.distance/(train.trip_duration/3600) > 60
train = train[~mask]
train.trip_duration = train.trip_duration.astype(np.uint16)
train = train[train.passenger_count > 0]

m = train.groupby(['wday','vendor_id'])[['trip_duration']].apply(np.median)
m.name = 'trip_duration_median'
train = train.join(m, on=['wday','vendor_id'])

################################################################################################################

#sns.lmplot(y='trip_duration_median', x='wday',data=train, fit_reg=False, hue='vendor_id')

################################################################################################################

m2 = train.groupby(['pu_hour','vendor_id'])[['trip_duration']].apply(np.median)
m2.name ='trip_duration_median_hour'
train = train.join(m2, on=['pu_hour','vendor_id'])

################################################################################################################

#sns.lmplot(y='trip_duration_median_hour', x='pu_hour',data=train, fit_reg=False, hue='vendor_id')

################################################################################################################

jfk_lon = -73.778889
jfk_lat = 40.639722
lga_lon = -73.872611
lga_lat = 40.77725

train['jfk_pickup_dist'] = train.apply(lambda row: haversine(jfk_lon, jfk_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)
train['lga_pickup_dist'] = train.apply(lambda row: haversine(lga_lon, lga_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)
train['jfk_dropoff_dist'] = train.apply(lambda row: haversine(jfk_lon, jfk_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)
train['lga_dropoff_dist'] = train.apply(lambda row: haversine(lga_lon, lga_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

train['jfk'] = ((train['jfk_pickup_dist'] < 2) | (train['jfk_dropoff_dist'] < 2))
train['lga'] = ((train['lga_pickup_dist'] < 2) | (train['lga_dropoff_dist'] < 2))
train = train.drop(['jfk_pickup_dist','lga_pickup_dist','jfk_dropoff_dist','lga_dropoff_dist'],axis=1)

train['workday'] = ((train['pu_hour'] > 8) & (train['pu_hour'] < 18))

################################################################################################################


#fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(10,8))
#plt.ylim(40.6, 40.9)
#plt.xlim(-74.1,-73.7)
#ax.scatter(train['pickup_longitude'],train['pickup_latitude'], s=0.01, alpha=1)

################################################################################################################

# RMSLE: Evaluation Metric
def rmsle(evaluator,X,real):
    sum = 0.0
    predicted = evaluator.predict(X)
    print("Number predicted less than 0: {}".format(np.where(predicted < 0)[0].shape))

    predicted[predicted < 0] = 0
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p-r)**2
    return (sum/len(predicted))**0.5


################################################################################################################

test.pickup_datetime=pd.to_datetime(test.pickup_datetime)
#test.dropoff_datetime=pd.to_datetime(test.dropoff_datetime)
test['pu_hour'] = test.pickup_datetime.dt.hour
test['yday'] = test.pickup_datetime.dt.dayofyear
test['wday'] = test.pickup_datetime.dt.dayofweek
test['month'] = test.pickup_datetime.dt.month
test['distance'] = haversine(test.pickup_longitude, test.pickup_latitude, test.dropoff_longitude, test.dropoff_latitude)
test = pd.merge(test,fastest_routes_test,on='id',how='outer')
test = test.drop(['step_location_list','step_direction','step_maneuvers','travel_time_per_step','distance_per_step','street_for_each_step','number_of_steps','starting_street','end_street'],axis=1)
test = pd.merge(test,weather_data,on='yday')
test = test.drop(['date','maximum temperature','minimum temperature'],axis=1)
test['jfk_pickup_dist'] = test.apply(lambda row: haversine(jfk_lon, jfk_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)
test['lga_pickup_dist'] = test.apply(lambda row: haversine(lga_lon, lga_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)
test['jfk_dropoff_dist'] = test.apply(lambda row: haversine(jfk_lon, jfk_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)
test['lga_dropoff_dist'] = test.apply(lambda row: haversine(lga_lon, lga_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)
test['jfk'] = ((test['jfk_pickup_dist'] < 2) | (test['jfk_dropoff_dist'] < 2))
test['lga'] = ((test['lga_pickup_dist'] < 2) | (test['lga_dropoff_dist'] < 2))
test = test.drop(['jfk_pickup_dist','lga_pickup_dist','jfk_dropoff_dist','lga_dropoff_dist'],axis=1)
test['workday'] = ((test['pu_hour'] > 8) & (test['pu_hour'] < 18))


################################################################################################################

# Benchmark Model
benchmark = fastest_routes_test[['id','total_travel_time']]
benchmark = benchmark.rename(index=str, columns={"total_travel_time": "trip_duration"})
#benchmark.head()

benchmark['trip_duration'].isnull().values.any()
benchmark.to_csv('benchmark.csv',index=False)
#RMSLE=0.990

################################################################################################################

features = train[['vendor_id','passenger_count','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pu_hour','wday','month','workday','precipitation','snow fall','snow depth','total_distance','total_travel_time','jfk','lga']]
target = train['trip_duration']

####################################################################################################################

testfeatures = test[['vendor_id','passenger_count','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pu_hour','wday','month','workday','precipitation','snow fall','snow depth','total_distance','total_travel_time','jfk','lga']]

####################################################################################################################
testfeatures.info()
print('Preprocessing complete.')
####################################################################################################################

# Linear Regression



lreg = LinearRegression()
cv = ShuffleSplit(n_splits=9)
print(cross_val_score(lreg, features, np.ravel(target), cv=cv, scoring=rmsle))
lreg.fit(features, target)

np.mean([0.43999617,  0.44022755 , 0.43897449,0.44073678])

testfeatures.shape

#ask
pred = lreg.predict(testfeatures)
print(np.where(pred < 0)[0].shape)
pred[pred < 0]=0

test['trip_duration']=pred.astype(int)
outlreg = test[['id','trip_duration']]

outlreg['trip_duration'].isnull().values.any()

outlreg.to_csv('pred_linear.csv',index=False)

# K-nearest Neighbors Regression


#KNNR = KNeighborsRegressor(n_neighbors=10)
#cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
#print(cross_val_score(KNNR, features, np.ravel(target), cv=cv,scoring=rmsle))
#KNNR.fit(features,target)
knr = KNeighborsRegressor()
parameters = [{'n_neighbors' : [5, 15, 35], 'weights' : ['uniform', 'distance'], 'algorithm' : ['auto'], 'leaf_size' : [30, 90, 150] }]

grid = GridSearchCV(estimator=knr, param_grid = parameters, cv = 9, n_jobs=-1)

grid.fit(features, np.ravel(target))
grid.best_score_

grid.best_params_

#{'algorithm': 'auto',
 #'leaf_size': 30,
 #'n_neighbors': 35,
 #'weights': 'distance'}

np.mean([0.42010954, 0.41940803, 0.41947931, 0.41968818])

pred = KNNR.predict(testfeatures)
print(np.where(pred < 0)[0].shape)

test['trip_duration']=pred.astype(int)
outKNNR = test[['id','trip_duration']]

outKNNR.to_csv('pred_knnr.csv',index=False)

from sklearn.ensemble import RandomForestRegressor

RandomForestRegressor()

# Random Forest
#ask

rfr = RandomForestRegressor()
#cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
#print(cross_val_score(rfr, features, np.ravel(target), cv=cv,scoring=rmsle))
#rfr = rfr.fit(features,np.ravel(target))

# 'max_leaf_nodes':[None], 'min_samples_leaf':[1], 'min_samples_split':[2],
rfrparameters = [{'max_depth':[10], 'n_estimators':[250], 'random_state':[42]}]

rfrgrid = GridSearchCV(estimator=rfr, param_grid = rfrparameters, cv = 7)

rfrgrid.fit(features, np.ravel(target))
rfrgrid.best_score_

# 0.5704935986667378

grid.best_params_

pred = rfrgrid.predict(testfeatures)
print(np.where(pred < 0)[0].shape)

test['trip_duration']=pred.astype(int)
outrfr = test[['id','trip_duration']]

outrfr.to_csv('pred_rfr2.csv',index=False)

xgb.XGBRegressor()

# XGBoost

# xgbreg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

# cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
# print(cross_val_score(xgbreg, features, np.ravel(target), cv=cv,scoring=rmsle))
# xgbreg.fit(features,target)

#xgbreg = xgb.XGBRegressor(n_estimators=100, seed=0,learning_rate=0.1, subsample=0.8,
#                           colsample_bytree=1, max_depth=7,min_child_weight= 1)

#cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
#print(cross_val_score(reg, features, np.ravel(target), cv=cv,scoring=rmsle))
#xgbreg.fit(features,target)


xgbreg = xgb.XGBRegressor()

#parameters = [{'base_score':[0.5], 'booster':['gbtree'], 'colsample_bylevel':[1],
#             'colsample_bynode':[1], 'colsample_bytree':[1], 'gamma':[0, 5],
#             'importance_type':['gain'], 'learning_rate':[0.1], 'max_delta_step':[0],
#             'max_depth':[3, 10], 'min_child_weight':[1, 5], 'missing':[None], 'n_estimators':[150, 250],
#             'n_jobs':[1], 'nthread':[None], 'objective':['reg:linear'], 'random_state':[42],
#             'reg_alpha':[0], 'reg_lambda':[1], 'scale_pos_weight':[1], 'seed':[None],
#             'silent':[None], 'subsample':[1], 'verbosity':[1]}]

parameters = [{'objective':['reg:squarederror'], 'colsample_bytree':[0.5], 'gamma':[5], 'max_depth':[10], 'n_estimators':[250], 'random_state':[42]}]



grid = GridSearchCV(estimator=xgbreg, param_grid = parameters, cv = 7)

grid.fit(features, np.ravel(target))
grid.best_score_

#0.6041168781346093
#0.6030209212995533

grid.best_params_

pred = grid.predict(testfeatures)
print(np.where(pred < 0)[0].shape)

pred[pred < 0] = 0
test['trip_duration']=pred.astype(int)
outxgbreg = test[['id','trip_duration']]
outxgbreg['trip_duration'].isnull().values.any()

outxgbreg.to_csv('pred_xgboost-gs3.csv',index=False)
print('completed')

from xgboost import plot_tree
plot_tree(reg)

import pickle
pickle.dump(reg, open('xgb_model.sav','wb'),protocol=2)

list(features.columns)

features.shape

features.head()

mlp = MLPRegressor()

mlp.fit(features, target.values.ravel())
predicted = mlp.predict(testfeatures)

print(np.where(predicted < 0)[0].shape)

predicted[predicted < 0] = 0
test['trip_duration']=predicted.astype(int)
outmlp = test[['id','trip_duration']]
outmlp['trip_duration'].isnull().values.any()

outmlp.to_csv('pred_mlp1.csv',index=False)
print('completed')


#expected = Y_test

#accuracy_score(expected,predicted)

# 0.2627324171382377


#parameters = {
#    'hidden_layer_sizes': [(10,), (50,), (100,)],
#    'activation': ['relu', 'tanh', 'logistic'],
#    'learning_rate': ['constant', 'invscaling', 'adaptive']
#}

#mlpgs = GridSearchCV(mlp, parameters, cv=5)
#mlpgs.fit(X_train, Y_train.values.ravel())