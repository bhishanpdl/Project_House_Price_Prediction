#!/usr/bin/env python

__doc__ = """
Author: Bhishan Poudel

Task: Regression modelling of King Country Seattle house price estimation.

Model used: Random forest with n_estimators = 49
  

adjusted r-squared
-------------------
num + nologs + cats: 0.886 (plain)
num + nologs + cats_encoded : 0.883  (ENCODING IS BAD)
num + nologs + cats_age + cats_agernv: 0.847
num + nologs + cats_age + cats_agernv + cats :0.885

"""

# Imports
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


def remove_outliers(df):
    df = df.drop(df[df["bedrooms"]>=10].index )
    df = df.drop(df[df["bathrooms"]>=7].index )
    df = df.drop(df[df["grade"].isin([3,1])].index )
    
    # we must reset index after removing outliers
    df = df.reset_index(drop=True)
    return df


def standard_scaling(df):
    from sklearn.preprocessing import StandardScaler

    ss = StandardScaler()
    array_scaled_feat = ss.fit_transform(df.drop('price',axis=1))
    df_feat = pd.DataFrame(array_scaled_feat,
                           columns = df.drop('price',axis=1).columns)
    df = pd.concat([df_feat, df[target]], axis=1)

    return df


def adjustedR2(rsquared,nrows,kcols):
    return rsquared- (kcols-1)/(nrows-kcols) * (1-rsquared)


def multiple_linear_regression(df,features,target,model,
                               verbose=1,cv=5,test_size=0.3):
    """ Multiple Linear Regression Modelling using given model.
    
    Depends:
    Depends on function adjusted r-squared.
    
    
    Returns:
    rmse, r2_train, ar2_train, r2_test, ar2_test, cv
    """
    
    # train test split
    train, test = train_test_split(df, test_size=test_size, random_state=100)

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
    
    return (rmse, r2_train, ar2_train, r2_test, ar2_test, cv)


df_eval = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})
#-----------------------------------------------------------------------------
if __name__ == '__main__':

    t0 = time.time()
     
    # load the data
    df = pd.read_csv('../data/processed/data_cleaned_encoded.csv')
    
    
    target = ['price']

    # plain features
    features_num = ['bedrooms', 'bathrooms',  'yr_built', 'lat', 'long']
    features_cat = ['waterfront', 'view', 'condition', 'grade','zipcode']
    features_no_log = ['sqft_living','sqft_lot','sqft_above',
                       'sqft_basement','sqft_living15','sqft_lot15']
    
    
    # log
    features_log = ['log1p_sqft_living','log1p_sqft_lot',
                    'log1p_sqft_above','log1p_sqft_basement',
                    'log1p_sqft_living15','log1p_sqft_lot15']

    # categorical encoding
    features_cat_age = [ 'age_cat_0', 'age_cat_1', 'age_cat_2',
                         'age_cat_3', 'age_cat_4', 'age_cat_5',
                         'age_cat_6', 'age_cat_7', 'age_cat_8',
                         'age_cat_9']

    feature_cat_agernv = [
                    'age_after_renovation_cat_0','age_after_renovation_cat_1',
                    'age_after_renovation_cat_2', 'age_after_renovation_cat_3',
                    'age_after_renovation_cat_4', 'age_after_renovation_cat_5',
                    'age_after_renovation_cat_6', 'age_after_renovation_cat_7',
                    'age_after_renovation_cat_8', 'age_after_renovation_cat_9']
    
    # newly created boolean features
    features_bool = ['basement_bool', 'renovation_bool']
    
    # newly created number of houses in given zipcode
    features_zipcode_extra = ['zipcode_houses']
    
    
    # all categorical features encoded.
    features_cat_encoded = [
        # waterfront
        'waterfront_0', 'waterfront_1',
        # view
        'view_0', 'view_1', 'view_2','view_3','view_4',
        # condition
        'condition_1', 'condition_2','condition_3', 'condition_4',
        'condition_5',
        # grade
        'grade_1', 'grade_10', 'grade_11', 'grade_12',
        'grade_13', 'grade_3', 'grade_4', 'grade_5', 'grade_6', 'grade_7',
        'grade_8', 'grade_9',
        # zipcode
        'zipcode_top10_98004', 'zipcode_top10_98006','zipcode_top10_98033',
        'zipcode_top10_98039', 'zipcode_top10_98040','zipcode_top10_98102',
        'zipcode_top10_98105', 'zipcode_top10_98155','zipcode_top10_98177',
        # age
        'age_cat_0', 'age_cat_1', 'age_cat_2','age_cat_3', 'age_cat_4',
        'age_cat_5', 'age_cat_6', 'age_cat_7','age_cat_8', 'age_cat_9',
        # age after renovation
        'age_after_renovation_cat_0',
        'age_after_renovation_cat_1', 'age_after_renovation_cat_2',
        'age_after_renovation_cat_3', 'age_after_renovation_cat_4',
        'age_after_renovation_cat_5', 'age_after_renovation_cat_6',
        'age_after_renovation_cat_7', 'age_after_renovation_cat_8',
        'age_after_renovation_cat_9']

    
    features = features_num + features_no_log + features_cat
    df = df[features + target]
    
    # options
    use_scaling = True
    use_remove_outliers = False
  
    text = "use_scaling = {}, remove_outliers = {} ".format(
        use_scaling, use_remove_outliers)
        
    if use_scaling:
        df = standard_scaling(df)
        
    if use_remove_outliers:
        df = remove_outliers(df)

    model = RandomForestRegressor(n_estimators= 50,random_state=100)
    rmse, r2_train, ar2_train, r2_test, ar2_test, cv = \
        multiple_linear_regression(df, features, target,model,
                                   verbose=0,test_size=0.2)


    df_eval.loc[len(df_eval)] = ['Random Forest Regressor',
                                 text, rmse,r2_train,ar2_train,
                                 r2_test,ar2_test,cv]
    
    for k,v in df_eval.to_dict().items():
        print(k, ':', v)

    t1 = time.time() - t0
    print('\n\nTime taken: {:.0f} min {:.0f} secs'.format(*divmod(t1,60)))