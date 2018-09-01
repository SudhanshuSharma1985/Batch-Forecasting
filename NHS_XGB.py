# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:02:39 2018

@author: sudhanshu sharma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import math
from sklearn.grid_search import GridSearchCV
import datetime as dt
import datetime


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


##Read demand data
df_train = pd.read_csv(r'C:\Sudhanshu\CurrentProj\PoornaData.csv')

##read BG demand data
df_train_BG = pd.read_csv(r'C:\Sudhanshu\CurrentProj\BG_output_future.csv')

df_train_BG =  df_train_BG[['Date','skill_id','BG_actual_demand','BG_Predicted_demand']]

##read GT demand data
df_train_GT = pd.read_csv(r'C:\Sudhanshu\CurrentProj\GT_output_future.csv')


df_train_BG['Date'] =  pd.to_datetime(df_train_BG['Date'])
df_train_GT['Date'] =  pd.to_datetime(df_train_GT['Date'])

df_train_BG1 = df_train_BG[(df_train_BG['Date'] >= datetime.datetime(2012,9,1)) & (df_train_BG['Date'] <= datetime.datetime(2018,2,1))]
df_train_BG2 = df_train_BG[~((df_train_BG['Date'] >= datetime.datetime(2012,9,1)) & (df_train_BG['Date'] <= datetime.datetime(2018,2,1)))]
df_train_BG2['demand'] =  np.nan

df_train = df_train.rename(columns={'skill_final': 'skill_id'})
#df_train = df_train.assign(Date=pd.to_datetime(df_train[['Year','Month']].assign(day=1)))
df_train['Date'] = pd.to_datetime(df_train['Date'])

## combine BG and demand data
df_train_combined = pd.merge(df_train_BG1,df_train[['skill_id','Date','demand']],on=['skill_id','Date'],how='inner')
df_train_combined_F = pd.concat([df_train_combined,df_train_BG2])

#merge Google trends and demand data

df_train_combined_OA = pd.merge(df_train_combined_F,df_train_GT[['skill_id','Date','GT_ActualDemand','GT_predictedDemand']],on=['skill_id','Date'],how='inner')

df_train_combined_OA['skill_id'].nunique()

df_train_combined_OA['Date'] = df_train_combined_OA['Date'].astype(str)

df_train_combined_OA.dtypes

df_train_combined_OA['Year'] = df_train_combined_OA['Date'].apply(lambda x: x.split('-')[0])
df_train_combined_OA['Month'] = df_train_combined_OA['Date'].apply(lambda x: x.split('-')[1])

df_train_combined_OA['Date'] =  pd.to_datetime(df_train_combined_OA['Date'])
df_train_combined_OA['Year']  = df_train_combined_OA['Year'].astype(int)
df_train_combined_OA['Month']  = df_train_combined_OA['Month'].astype(int)


df_train.dtypes
df_train_combined_OA.dtypes
df_train_GT.set_index('Date', inplace=True)

df_train_combined2 = df_train_combined_OA 

df_train_combined2.dtypes

df_train_combined2['Date']  = df_train_combined2['Date'].astype(str)

df_train_combined2.set_index(pd.DatetimeIndex(df_train_combined2['Date']), inplace=True)

df_train_combined2_1=df_train_combined2.ix['2018-01-01']


df_train_combined2_1 = df_train_combined2_1.drop(['Date'],axis=1)
df_train_combined2_1 =  df_train_combined2_1.reset_index()
df_train_combined2 = df_train_combined2.drop(['Date'],axis=1)
df_train_combined2 =  df_train_combined2.reset_index()

df_train_combined2_1['demand'] = target_test2['Predictions']

df_train_combined2.dtypes
df_train_combined2_1.dtypes

df_train_combined2_1['Date']  = df_train_combined2_1['Date'].astype(str)
df_train_combined2['Date']  = df_train_combined2['Date'].astype(str)

df_train_combined2_2 = pd.merge(df_train_combined2,df_train_combined2_1[['demand','skill_id','Date']],on=['skill_id','Date'],how='left')

df_train_combined2_2['demand_y'].fillna(df_train_combined2_2['demand_x'], inplace=True)

df_train_combined2_2 = df_train_combined2_2.rename(columns={'demand_y': 'demand'})

df_train_combined4 = []

## loop through each skill and create lag variables for all 3 type of demands - Within company demand, BG demand and Google trends demand

for i in df_train_combined2.skill_id.unique():
    df_train_combined3 = df_train_combined2[df_train_combined2['skill_id'] == i]
    df_train_combined3['demand_lag1']= df_train_combined3['demand'].shift(1)
    df_train_combined3['demand_lag2']= df_train_combined3['demand'].shift(2)
    df_train_combined3['demand_lag3']= df_train_combined3['demand'].shift(3)
    df_train_combined3['demand_lag4']= df_train_combined3['demand'].shift(4)
    df_train_combined3['demand_lag5']= df_train_combined3['demand'].shift(5)
    df_train_combined3['demand_lag6']= df_train_combined3['demand'].shift(6)
    df_train_combined3['demand_lag7']= df_train_combined3['demand'].shift(7)
    df_train_combined3['demand_lag8']= df_train_combined3['demand'].shift(8)
    df_train_combined3['demand_lag9']= df_train_combined3['demand'].shift(9)
    df_train_combined3['demand_lag10']= df_train_combined3['demand'].shift(10)
    df_train_combined3['demand_lag11']= df_train_combined3['demand'].shift(11)
    df_train_combined3['demand_lag12']= df_train_combined3['demand'].shift(12)
    
    df_train_combined3['BG_demand_lag1']= df_train_combined3['BG_actual_demand'].shift(1)
    df_train_combined3['BG_demand_lag2']= df_train_combined3['BG_actual_demand'].shift(2)
    df_train_combined3['BG_demand_lag3']= df_train_combined3['BG_actual_demand'].shift(3)
    df_train_combined3['BG_demand_lag4']= df_train_combined3['BG_actual_demand'].shift(4)
    df_train_combined3['BG_demand_lag5']= df_train_combined3['BG_actual_demand'].shift(5)
    df_train_combined3['BG_demand_lag6']= df_train_combined3['BG_actual_demand'].shift(6)
    df_train_combined3['BG_demand_lag7']= df_train_combined3['BG_actual_demand'].shift(7)
    df_train_combined3['BG_demand_lag8']= df_train_combined3['BG_actual_demand'].shift(8)
    df_train_combined3['BG_demand_lag9']= df_train_combined3['BG_actual_demand'].shift(9)
    df_train_combined3['BG_demand_lag10']= df_train_combined3['BG_actual_demand'].shift(10)
    df_train_combined3['BG_demand_lag11']= df_train_combined3['BG_actual_demand'].shift(11)
    df_train_combined3['BG_demand_lag12']= df_train_combined3['BG_actual_demand'].shift(12)
    
    df_train_combined3['GT_demand_lag1']= df_train_combined3['GT_ActualDemand'].shift(1)
    df_train_combined3['GT_demand_lag2']= df_train_combined3['GT_ActualDemand'].shift(2)
    df_train_combined3['GT_demand_lag3']= df_train_combined3['GT_ActualDemand'].shift(3)
    df_train_combined3['GT_demand_lag4']= df_train_combined3['GT_ActualDemand'].shift(4)
    df_train_combined3['GT_demand_lag5']= df_train_combined3['GT_ActualDemand'].shift(5)
    df_train_combined3['GT_demand_lag6']= df_train_combined3['GT_ActualDemand'].shift(6)
    df_train_combined3['GT_demand_lag7']= df_train_combined3['GT_ActualDemand'].shift(7)
    df_train_combined3['GT_demand_lag8']= df_train_combined3['GT_ActualDemand'].shift(8)
    df_train_combined3['GT_demand_lag9']= df_train_combined3['GT_ActualDemand'].shift(9)
    df_train_combined3['GT_demand_lag10']= df_train_combined3['GT_ActualDemand'].shift(10)
    df_train_combined3['GT_demand_lag11']= df_train_combined3['GT_ActualDemand'].shift(11)
    df_train_combined3['GT_demand_lag12']= df_train_combined3['GT_ActualDemand'].shift(12)
    df_train_combined4.append(df_train_combined3)
    
df_train_combined5 = pd.concat(df_train_combined4)
df_train_combined5 = df_train_combined5.drop(['Date'],axis=1)
df_train_combined5 =  df_train_combined5.reset_index()


df = df_train_combined5['skill_id']
df_train_combined5.skill_id.nunique()
df_train_combined5 = df_train_combined5.reset_index()

## remove those skills where demand data is less than 30 months
demand_count = df_train_combined5.groupby(['skill_id'])[['Date']].count()
demand_count = demand_count[demand_count['Date'] >=30]
demand_count = demand_count.reset_index()

df_train_combined6 = pd.merge(df_train_combined5,demand_count[['skill_id']],on=['skill_id'],how='inner')
df_train_combined6.skill_id.nunique()
df_train_combined6.dtypes

df_train_combined6['Date'] =  df_train_combined6['Date'].astype(str)

##create new features - Year and month

df_train_combined6['time_year'] = df_train_combined6['Date'].apply(lambda x: x.split('-')[0])
df_train_combined6['time_month'] = df_train_combined6['Date'].apply(lambda x: x.split('-')[1])
#df_train_combined6 = df_train_combined6.rename(columns={'time_year': 'Year','time_month': 'Month','Skill ID from MST': 'skill_final','Demand': 'BG_Demand'})


##removing NA values
df_train_combined6.fillna(0, inplace=True)

df_train_combined6.dtypes

df_train_combined6['Date'] = pd.to_datetime(df_train_combined6['Date'])
df_train_combined6['Year'] = df_train_combined6['Year'].astype(int)
df_train_combined6['Month'] = df_train_combined6['Month'].astype(int)  


target = df_train_combined6[['demand','Date','skill_id']]



features = df_train_combined6.drop(['demand','BG_Predicted_demand','GT_predictedDemand'],axis=1)
        
from sklearn.model_selection import train_test_split

## Split the data into train and test
        
target.set_index(pd.DatetimeIndex(target['Date']), inplace=True)
target_train=target.ix['2012-09-01':'2018-01-01']
target_test = target.ix['2018-02-01']
target_test = target_test.drop(['Date'],axis=1)
target_test =  target_test.reset_index()

features.set_index(pd.DatetimeIndex(features['Date']), inplace=True)
features_train=features.ix['2012-09-01':'2018-01-01']
features_train = features_train.drop(['Date'],axis=1)

features_test=features.ix['2018-02-01']
features_test = features_test.drop(['Date'],axis=1)
    
target_train = target_train.drop(['Date'],axis=1)
target_test = target_test.drop(['Date'],axis=1)
    
target_test4 = target_test['skill_id']
target_test4 =  target_test4.reset_index()

target_test5 = target_train['skill_id']
target_test5 =  target_test5.reset_index()

    
features_train["skill_id"] = features_train["skill_id"].astype('category')
features_train["skill_id"] = features_train["skill_id"].cat.codes
    
features_test["skill_id"] = features_test["skill_id"].astype('category')
features_test["skill_id"] = features_test["skill_id"].cat.codes
features_test.dtypes

    
target_train = target_train.drop(['skill_id'],axis=1)
target_test = target_test.drop(['skill_id'],axis=1)

features_train = features_train.drop(['time_year'],axis=1)
features_train = features_train.drop(['time_month'],axis=1)

features_test = features_test.drop(['time_year'],axis=1)
features_test = features_test.drop(['time_month'],axis=1)

    
import xgboost as xgb
features_train.dtypes

features_train = features_train.drop(['index'],axis=1)
features_test = features_test.drop(['index'],axis=1)

DM_train = xgb.DMatrix(data = features_train, label = target_train )
#target_test = target_test.drop(['Index'],axis=1)
DM_test = xgb.DMatrix(data = features_test, label = target_test  )

# train the model
        
params = {"booster":"gbtree","objective":"reg:linear", "eta":"0.1", "max_depth":"20", "colsample_bytree":"0.8"}
xg_reg = xgb.train(params = params, dtrain = DM_train, num_boost_round = 2000)
    
    
#parameter tuning
    
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10, 100]},{'learning_rate': [0.1, 0.01, 0.5]}]
gbm=xgb.XGBClassifier(max_features='sqrt', subsample=0.8, random_state=10)
grid_search = GridSearchCV(estimator = gbm, param_grid = parameters, scoring='accuracy', cv = 3, n_jobs=-1)
grid_search = grid_search.fit(features_train, target_train)
warnings.filterwarnings("ignore")

grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_

grid_search.best_estimator_

xg_reg=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, max_features='sqrt', min_child_weight=1, missing=None,
       n_estimators=10, n_jobs=1, nthread=None,
       objective='multi:softprob', random_state=10, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=0.8).fit(features_train, target_train)
    
    
xgb.plot_importance(xg_reg)
plt.show()

preds = xg_reg.predict(DM_test)
    
preds = pd.DataFrame(preds)

preds = preds.rename(columns={'O': 'Prediction'})
Pred1 = preds
Pred1['Prediction'] = preds
Pred2 = Pred1['Prediction']
preds = Pred2
preds = pd.DataFrame(preds)
preds.columns
preds['skill_id'] = target_test4['skill_id']

target_test2 = pd.DataFrame(target_test)
target_test2 =  target_test2.reset_index()
target_test2['Predictions'] = preds['Prediction']
target_test2['skill_id'] = preds['skill_id']
target_test2.to_csv('TargetsVsPred_30.csv')

##ENDS HERE


