# Import libraries
import time
time_start_notebook = time.time()
import numpy as np
import pandas as pd
import argparse

# local imports
import config
import util

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

# special
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#===================== parameters
ifile = config.data_path
params_rf = config.params_rf
target = config.target
train_size = config.train_size
cols_drop = config.cols_drop
cols_log = config.cols_log

params_xgb = config.params_xgb
params_lgb = config.params_lgb
params_cb = config.params_cb

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-name','--name',help='name of booster eg. xgb lgb cb',type=str,required=True)
args = parser.parse_args()
name = args.name

# boosting model
if name == 'xgb':
    model = XGBRegressor(**params_xgb)
elif name == 'lgb':
    model = LGBMRegressor(**params_lgb)
elif name == 'cb':
    model = CatBoostRegressor(**params_cb)
else:
    print('Please use one of boosting model: xgb, lgb, cb')
    sys.exit(1)

#=================== load the data
df = pd.read_csv(ifile)

#========================== data processing
df = df.drop(cols_drop,axis=1)
for col in cols_log:
    df[col] = np.log1p(df[col].to_numpy())

#======================== train test split
df_Xtrain,df_Xtest,ser_ytrain,ser_ytest = train_test_split(
    df.drop([target],axis=1),
    df[target],
    train_size=train_size,
    random_state=SEED)

ytrain = np.array(ser_ytrain).flatten()
ytest = np.array(ser_ytest).flatten()

#============================= scaling
scaler = preprocessing.StandardScaler()
scaler.fit(df_Xtrain)
Xtrain = scaler.transform(df_Xtrain)
Xtest  = scaler.transform(df_Xtest)

#===================== modelling
model.fit(df_Xtrain,ser_ytrain)

#======================= model evaluation
# NOTE: we need to do inverse log transform of target
ypreds = model.predict(df_Xtest).flatten()
ytest = np.expm1(ytest)
ypreds = np.expm1(ypreds)
util.print_regr_eval(ytest,ypreds,ncols=df_Xtest.shape[1])

#========================= time taken
time_taken = time.time() - time_start_notebook
util.print_time_taken(time_taken)


# results
res = """
             RMSE : 113,922.65
         R-Squared: 0.9037
Adjusted R-squared: 0.9033

Time Taken: 14.03 sec sec
"""