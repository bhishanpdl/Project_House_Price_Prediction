#!/usr/bin/env python

__doc__ = """
Author: Bhishan Poudel

Task
-------------------
Regression modelling of King Country Seattle house price estimation.

Model used
-------------------------
Random forest 
n_estimators = 50
max_depth = 50
topN features = 40

Result:
---------------------------
Adjusted R-Squared (test): 0.890


"""
#=============================================================================
# Imports
#=============================================================================
import numpy as np
import pandas as pd

import os
import time
import collections
import itertools

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# random state
SEED = 0
RNG = np.random.RandomState(SEED)

#=============================================================================
# Utilities
#=============================================================================
def multiple_linear_regression(df,features,target,model,
                            verbose=1,cv=5,test_size=0.3):
    """ Multiple Linear Regression Modelling using given model.

    Depends:
    Depends on function adjusted r-squared.


    Returns:
    rmse, r2_train, ar2_train, r2_test, ar2_test, cv
    """
    def adjustedR2(rsquared,nrows,kcols):
        return rsquared- (kcols-1)/(nrows-kcols) * (1-rsquared)


    # train test split
    train, test = train_test_split(df, test_size=0.2, random_state=100)

    # train test values
    X = df[features].values
    y = df[target].values.ravel()

    Xtrain = train[features].values
    ytrain = train[target].values.ravel()

    Xtest = test[features].values
    ytest = test[target].values.ravel()

    # fitting
    model.fit(Xtrain,ytrain)

    # prediction
    ypreds = model.predict(Xtest)

    # metrics
    rmse = np.sqrt(mean_squared_error(ytest,ypreds)).round(3)
    r2_train = model.score(Xtrain, ytrain).round(3)
    r2_test = model.score(Xtest, ytest).round(3)

    cv = cross_val_score(model, X, y, cv=5,n_jobs=-1,
                         verbose=verbose).mean().round(3)

    ar2_train = adjustedR2(model.score(Xtrain,ytrain),
                           Xtrain.shape[0],
                           len(features)).round(3)
    ar2_test  = adjustedR2(model.score(Xtest,ytest),
                           Xtest.shape[0] ,
                           len(features)).round(3)
    
    return (model, rmse, r2_train, ar2_train, r2_test, ar2_test, cv)


df_eval = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})

t0 = time.time()
#=============================================================================
# Data Loading
#=============================================================================
# load the data
df_raw = pd.read_csv('../data/processed/data_cleaned_encoded.csv')

#=============================================================================
# Train test split
#=============================================================================
# train test split
train, test = train_test_split(df_raw,train_size = 0.8,random_state=RNG)

#=============================================================================
# Feature Selection
#=============================================================================
# feature selection
features_orig = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']

cols_num = ['bedrooms', 'bathrooms',
            'sqft_living', 'sqft_lot','sqft_above','sqft_basement',
            'yr_built', 'yr_renovated',
           'lat','long',
           'sqft_living15', 'sqft_lot15', 'yr_sales']

cols_bool = ['basement_bool', 'renovation_bool']

cols_new = ['zipcode_houses']

cols_cat = [
    # waterfront
    'waterfront_0', 'waterfront_1',
    
    #view
    'view_0', 'view_1', 'view_2', 'view_3','view_4',
    
    # condition
    'condition_1', 'condition_2', 'condition_3',
    'condition_4','condition_5',
    
    # grade
    'grade_1', 'grade_10', 'grade_11', 'grade_12','grade_13',
    'grade_3', 'grade_4', 'grade_5', 'grade_6', 'grade_7','grade_8', 'grade_9',
            
    # zipcode
    'zipcode_top10_98004', 'zipcode_top10_98006',
    'zipcode_top10_98033', 'zipcode_top10_98039',
    'zipcode_top10_98040','zipcode_top10_98102',
    'zipcode_top10_98105', 'zipcode_top10_98155',
    'zipcode_top10_98177']



cols_cat_age = [ 'age_cat_0', 'age_cat_1', 'age_cat_2',
                               'age_cat_3', 'age_cat_4', 'age_cat_5',
                               'age_cat_6', 'age_cat_7', 'age_cat_8',
                               'age_cat_9']

cols_cat_agernv = [
                'age_after_renovation_cat_0','age_after_renovation_cat_1',
                'age_after_renovation_cat_2', 'age_after_renovation_cat_3',
                'age_after_renovation_cat_4', 'age_after_renovation_cat_5',
                'age_after_renovation_cat_6', 'age_after_renovation_cat_7',
                'age_after_renovation_cat_8', 'age_after_renovation_cat_9']

features_all_encoded = cols_num + cols_bool + cols_new + cols_cat + cols_cat_age + cols_cat_agernv
target = ['price']

#=============================================================================
# Random Forest All encoded features after grid search best model
#=============================================================================
target = ['price']
features = features_all_encoded
df = df_raw[features + target]

model = RandomForestRegressor(n_estimators= 40,random_state=random_state,
                              max_features=69,
                              max_depth=50, bootstrap=True)

fitted_model, rmse, r2_train, ar2_train, r2_test, ar2_test, cv = \
    multiple_linear_regression(df, features, target,model,
                               verbose=2,test_size=0.2)


df_eval.loc[len(df_eval)] = ['Random Forest Regressor after grid search',
                             ' all encoded features, best grid search,\
                             n_estimators=40, max_features=69, max_depth=50',
                             rmse,r2_train,ar2_train,
                             r2_test,ar2_test,cv]

#=============================================================================
# Random Forest Feature Importance
#=============================================================================
importances = fitted_model.feature_importances_
df_imp = pd.DataFrame({'feature': features, 'importance': importances})
topN = 40
top_cols = df_imp.head(topN)['feature'].values.tolist()

features = top_cols
target = ['price']

df = df_raw[features + target]

model = RandomForestRegressor(n_estimators= 50,random_state=random_state,
                              max_depth=50, bootstrap=True)

fitted_model, rmse, r2_train, ar2_train, r2_test, ar2_test, cv = \
    multiple_linear_regression(df, features, target,model,
                               verbose=2,test_size=0.2)

df_eval.loc[len(df_eval)] = ['Random Forest Regressor',
                             'n_estimators = 50, max_depth = 50,\
                             topN features = '+str(topN),
                             rmse,r2_train,ar2_train,
                             r2_test,ar2_test,cv]

#=============================================================================
# Print Results
#=============================================================================
print('Features used:\n', df.columns.values)
print()
for k,v in df_eval.to_dict().items():
    print(k, ':', v)

t1 = time.time() - t0
print('\n\nTime taken: {:.0f} min {:.0f} secs'.format(*divmod(t1,60)))