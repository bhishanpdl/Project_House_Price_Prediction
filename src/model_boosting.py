# Import libraries
import time
time_start_notebook = time.time()
import numpy as np
import pandas as pd
import argparse

# local imports
import config
import util
from util import clean_data
from util import write_regr_eval

# random state
import os
import sys
import random
import numpy as np
SEED=config.SEED
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)

# machine learning
import functools
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# boosting
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn import ensemble # HistGradientBoostingRegressor
import xgboost  # XGBRegressor
import lightgbm # LGBMRegressor
import catboost # CatBoostRegressor

#===================== parameters
data_path_train = config.data_path_train
data_path_test = config.data_path_test

target = config.target
train_size = config.train_size

cols_drop = config.cols_drop
cols_sq = config.cols_sq

params_hgbr = config.params_hgbr
params_xgb = config.params_xgb
params_lgb = config.params_lgb
params_cb = config.params_cb

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-nm','--name',
                    help='name of booster eg. xgb lgb cb',
                    type=str,required=True)
parser.add_argument('-sc','--scaling',
                    help='eg. standard minmax',
                    type=str,
                    required=False)
parser.add_argument('-log','--log',
                    help='0 1',
                    type=int,
                    default=1,
                    required=False)
parser.add_argument('-sq','--sq',
                    help='0 1',
                    type=int,
                    default=1,
                    required=False)
parser.add_argument('-logsq','--logsq',
                    help='0 1',
                    type=int,
                    default=0,
                    required=False)
parser.add_argument('-logt','--logtarget',
                    help='0 1',
                    type=int,
                    default=1,
                    required=False)
parser.add_argument('-dummy','--dummy',
                    help='0 1',
                    type=int,
                    default=1,
                    required=False)
parser.add_argument('-dummycat','--dummycat',
                    help='0 1',
                    type=int,
                    default=0,
                    required=False)
parser.add_argument('-vb','--verbose',
                    help='0 1',
                    type=int,
                    default=1,
                    required=False)

args = parser.parse_args()
name = args.name
scaling = args.scaling
log = args.log
sq = args.sq
logsq = args.logsq
logtarget = args.logtarget
dummy = args.dummy
dummy_cat = args.dummycat
verbose = args.verbose

cb_cat = 0 # catboost cat_features (only use if we have not get dummies )

# print parameters
myargs = {'name'   : name,
        'scaling'  : scaling,
        'log'      : log,
        'sq'       : sq,
        'logsq'    : logsq,
        'logtarget': logtarget,
        'dummy'    : dummy,
        'dummy_cat': dummy_cat,
        'verbose'  : verbose
        }

#=================== load the data
df_train = pd.read_csv(data_path_train)
df_test = pd.read_csv(data_path_test)

#========================== data processing
df_train = clean_data(df_train,log=log,sq=sq,logsq=logsq,dummy=True,dummy_cat=False)
df_test = clean_data(df_test,log=log,sq=sq,logsq=logsq,dummy=True,dummy_cat=False)

#======================== feature selection
features = list(sorted(df_train.columns.drop(target)))
features = [i for i in features
            if i in df_test.columns
            if i in df_train.columns]
df_Xtrain  = df_train[features]
ser_ytrain = df_train[target]

df_Xtest  = df_test[features]
ser_ytest = df_test[target]

ytrain = np.array(ser_ytrain).flatten()
ytest  = np.array(ser_ytest).flatten()
features_train = list(df_Xtrain.columns)

if logtarget == 1:
    ytrain = np.log1p(ytrain)

#===================== scaling
if scaling == 'standard':
    scaler = preprocessing.StandardScaler()
    scaler.fit(df_Xtrain)
    df_Xtrain = pd.DataFrame(scaler.transform(df_Xtrain),columns=df_Xtrain.columns)
    df_Xtest = pd.DataFrame(scaler.transform(df_Xtest),columns=df_Xtrain.columns)
elif scaling == 'minmax':
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df_Xtrain)
    df_Xtrain = pd.DataFrame(scaler.transform(df_Xtrain),columns=df_Xtest.columns)
    df_Xtest = pd.DataFrame(scaler.transform(df_Xtest),columns=df_Xtest.columns)

#=================================== boosting model
if name == 'hgbr':
    model = ensemble.HistGradientBoostingRegressor(**params_hgbr)
if name == 'xgb':
    model = xgboost.XGBRegressor(**params_xgb)
elif name == 'lgb':
    model = lightgbm.LGBMRegressor(**params_lgb)
elif name == 'cb':
    if cb_cat:
        # if we have not already created dummy features, we can use cat_features
        # to use cat features in catboost the column must be int or str not float
        lst_cat_features = config.lst_cat_features
        for col in lst_cat_features:
            df_Xtrain[col] = df_Xtrain[col].astype(str)
            df_Xtest[col] = df_Xtest[col].astype(str)
        params_cb['cat_features'] = lst_cat_features

    model = catboost.CatBoostRegressor(**params_cb)

#===================== modelling
model.fit(df_Xtrain,ytrain)

#======================= model evaluation
ypreds = model.predict(df_Xtest).flatten()
if logtarget:
    ypreds = np.expm1(ypreds)

#========================= time taken
time_taken = time.time() - time_start_notebook
ncols = df_Xtest.shape[1]

# if verbost print additional info
if verbose:
    print(myargs)
    print('shape : ', df_train.shape)
    print(df_train.columns.to_numpy())
    print(model)
    util.print_regr_eval(ytest,ypreds,ncols)
    util.print_time_taken(time_taken)

# write results to a file
if not os.path.exists('outputs'):
    os.makedirs('outputs')

ofile = ('outputs/'   + name  +
        '_scaling_'   + str(scaling) +
        '_logsq_'     + str(logsq) +
        "_logtarget_" + str(logtarget)
        +'.csv')

util.write_regr_eval(ytest,ypreds,ncols,ofile)

# commands
command = """
py model_boosting --name xgb --scaling None --logsq 0

"""