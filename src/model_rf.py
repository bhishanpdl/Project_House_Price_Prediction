# Import libraries
import time
time_start_notebook = time.time()
import numpy as np
import pandas as pd

# local imports
import config
import util

# random state
import os
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
from sklearn import ensemble

#===================== parameters
data_path_clean = config.data_path_clean
target = config.target
train_size = config.train_size

params_rf = config.params_rf
cols_sq = config.cols_sq
cols_drop = config.cols_drop

#=================== load the data
df = pd.read_csv(data_path_clean)
df = df.drop(cols_drop, axis=1)

#========================== data processing
for col in cols_sq:
    df[col + '_sq'] = df[col]**2

#======================== train test split
df_Xtrain, df_Xtest, ser_ytrain, ser_ytest = train_test_split(
    df.drop([target],axis=1), df[target],
    train_size=train_size, random_state=SEED)

ytrain = np.array(ser_ytrain).flatten()
ytest = np.array(ser_ytest).flatten()

#===================== modelling
model = ensemble.RandomForestRegressor(**params_rf)
model.fit(df_Xtrain, ser_ytrain)

#======================= model evaluation
ypreds = model.predict(df_Xtest).flatten()
util.print_regr_eval(ytest,ypreds,ncols=df_Xtest.shape[1])

#========================= time taken
time_taken = time.time() - time_start_notebook
util.print_time_taken(time_taken)