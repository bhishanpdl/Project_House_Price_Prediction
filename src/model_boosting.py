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
data_path_clean = config.data_path_clean
target = config.target
train_size = config.train_size

cols_drop = config.cols_drop
cols_sq = config.cols_sq

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
df = pd.read_csv(data_path_clean)
df = df.drop(cols_drop, axis=1)

#========================== data processing
for col in cols_sq:
    df[col + '_sq'] = df[col]**2

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
ypreds = model.predict(df_Xtest).flatten()
util.print_regr_eval(ytest,ypreds,ncols=df_Xtest.shape[1])

#========================= time taken
time_taken = time.time() - time_start_notebook
util.print_time_taken(time_taken)


# results
res = """
$ py model_boosting.py --name xgb

             RMSE : 3,675.98
         R-Squared: 0.9999
Adjusted R-squared: 0.9999

Time Taken: 50.49 sec

$ py model_boosting.py --name lgb

             RMSE : 50,256.59
         R-Squared: 0.9813
Adjusted R-squared: 0.9808

Time Taken: 4.19 sec

$ py model_boosting.py --name cb

             RMSE : 52,299.88
         R-Squared: 0.9797
Adjusted R-squared: 0.9792

Time Taken: 51.76 sec
"""