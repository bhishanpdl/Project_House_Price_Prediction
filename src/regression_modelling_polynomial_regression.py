#!/usr/bin/env python

__doc__ = """
Author: Bhishan Poudel

Task: Regression modelling of King Country Seattle house price estimation.

Model used: Polynomial regression deg=2 only raw features

Result:
--------
Adjusted R-squared (test) :  0.813

"""
#=============================================================================
# Imports
#=============================================================================
import numpy as np
import pandas as pd
import os
import time

# random state
SEED = 0
RNG = np.random.RandomState(SEED)


# scale and split
from sklearn.model_selection import train_test_split

# regressors
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# regressor preprocessing
from sklearn.preprocessing import PolynomialFeatures

# metrics
from sklearn import metrics
from sklearn.metrics import mean_squared_error


features_raw_all = ['bedrooms','bathrooms','sqft_living','sqft_lot',
                    'floors','waterfront','view','condition','grade',
                    'sqft_above','yr_built','yr_renovated',
                    'zipcode','lat','long','sqft_living15','sqft_lot15']

# cross validation
from sklearn.model_selection import cross_val_score

df_eval = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})
#=============================================================================
# Data Loading
#=============================================================================
# load the data
df = pd.read_csv('../data/processed/data_cleaned_encoded.csv')

#=============================================================================
# Train test split
#=============================================================================

# train test split
train, test = train_test_split(df,train_size = 0.8,random_state=RNG)

#=============================================================================
# Feature Selection
#=============================================================================
# feature selection
target = ['price']
features_raw_all = ['bedrooms','bathrooms','sqft_living','sqft_lot',
                    'floors','waterfront','view','condition','grade',
                    'sqft_above','yr_built','yr_renovated',
                    'zipcode','lat','long','sqft_living15','sqft_lot15']

features = features_raw_all


polyfeat = PolynomialFeatures(degree=2)

X = polyfeat.fit_transform(df[features])

Xtrain = polyfeat.fit_transform(train[features])
Xtest = polyfeat.fit_transform(test[features])

y = df[target].values.reshape(-1,1)
ytrain = train[target].values.reshape(-1,1)
ytest = test[target].values.reshape(-1,1)


#=============================================================================
# Modelling
#=============================================================================
model = linear_model.LinearRegression(n_jobs=-1)

# modelling
def multiple_linear_regression(model,X,y, Xtrain, ytrain, Xtest,ytest,cv=5):
    """ Multiple Linear Regression Modelling using given model.
    
    
    Returns:
    rmse, r2_train, ar2_train, r2_test, ar2_test, cv
    """
    def adjustedR2(rsquared,nrows,kcols):
        return rsquared- (kcols-1)/(nrows-kcols) * (1-rsquared)
    
    # fitting
    model.fit(Xtrain,ytrain)

    # prediction
    ypreds = model.predict(Xtest)

    # metrics
    rmse = np.sqrt(mean_squared_error(ytest,ypreds)).round(3)
    r2_train = model.score(Xtrain, ytrain).round(3)
    r2_test = model.score(Xtest, ytest).round(3)

    cv = cross_val_score(model, X, y, cv=5,n_jobs=-1).mean().round(3)

    ar2_train = adjustedR2(model.score(Xtrain,ytrain),
                           Xtrain.shape[0],
                           len(features)).round(3)
    ar2_test  = adjustedR2(model.score(Xtest,ytest),
                           Xtest.shape[0] ,
                           len(features)).round(3)
    
    return (rmse, r2_train, ar2_train, r2_test, ar2_test, cv)

#=============================================================================
# Model Evaluation
#=============================================================================
rmse, r2_train, ar2_train, r2_test, ar2_test, cv = \
    multiple_linear_regression(model,X,y, Xtrain, ytrain, Xtest,ytest)


df_eval.loc[len(df_eval)] = ['Polynomial Regression','deg=2, all features,\
                              unprocessed, no regularization',
                             rmse,r2_train,ar2_train,r2_test,ar2_test,cv]

for k,v in df_eval.to_dict().items():
    print(k, ' : ', v[0])