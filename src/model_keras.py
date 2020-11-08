# Import libraries
import time
time_start_notebook = time.time()
import numpy as np
import pandas as pd

# mixed
from pprint import pprint
import os
import argparse

# machine learning
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# deep learning
import tensorflow as tf
import keras

# local imports
import util, util_keras,config,config_keras
from util import show_methods
from util import adjustedR2
from util import print_regr_eval
from util_keras import get_keras_model
from util_keras import set_random_seed
from util_keras import plot_keras_history
from util import clean_data

# params
data_path_train = config.data_path_train
data_path_test  = config.data_path_test
target          = config.target
train_size      = config.train_size

PARAMS_MODEL = config_keras.PARAMS_MODEL
METRICS      = config_keras.METRICS
PARAMS_FIT   = config_keras.PARAMS_FIT

# parse arguments
#===============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-nm','--name',
                    help='name of model eg. keras',
                    default='keras',
                    type=str,
                    required=False)
parser.add_argument('-sc','--scaling',
                    help='eg. standard minmax',
                    type=str,
                    default='standard',
                    required=False)
parser.add_argument('-log','--log',
                    help='0 1',
                    type=int,
                    default=1,
                    required=False)
parser.add_argument('-sq','--sq',
                    help='0 1',
                    type=int,
                    default=0,
                    required=False)
parser.add_argument('-logsq','--logsq',
                    help='0 1',
                    type=int,
                    default=0,
                    required=False)
parser.add_argument('-logt','--logtarget',
                    help='0 1',
                    type=int,
                    default=0,
                    required=False)
parser.add_argument('-dummy','--dummy',
                    help='0 1',
                    type=int,
                    default=0, # for keras do not use dummy
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

# print parameters
myargs = {
        'name'     : name,
        'scaling'  : scaling,
        'log'      : log,
        'sq'       : sq,
        'logsq'    : logsq,
        'logtarget': logtarget,
        'dummy'    : dummy,
        'dummy_cat': dummy_cat,
        'verbose'  : verbose
        }

if verbose:
    print()
    pprint(myargs)
    print()
#===============================================================================
# load the data
df_train = pd.read_csv(data_path_train)
df_test = pd.read_csv(data_path_test)


"""# Data Processing"""

#========================== data processing
data_params = dict(log=log,sq=sq,logsq=logsq,
                    dummy=dummy,dummy_cat=dummy_cat)
df_train = clean_data(df_train,**data_params)
df_test = clean_data(df_test,**data_params)


"""# Train Test Split"""

features = list(sorted(df_train.columns.drop(target)))
features = [i for i in features if i in df_test.columns
            if i in df_train.columns]

df_Xtrain  = df_train[features]
ser_ytrain = df_train[target]

df_Xtest  = df_test[features]
ser_ytest = df_test[target]

ytrain = np.array(ser_ytrain).flatten()
ytest  = np.array(ser_ytest).flatten()
if verbose:
    print('shape : ', df_train.shape)
    print(features)
    print()


"""# Scaling"""

scaling = 'standard'
if scaling == 'standard':
    scaler = preprocessing.StandardScaler()
    scaler.fit(df_Xtrain)
    Xtrain = scaler.transform(df_Xtrain)
    Xtest =  scaler.transform(df_Xtest)
elif scaling == 'minmax':
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df_Xtrain)
    Xtrain = scaler.transform(df_Xtrain)
    Xtest = scaler.transform(df_Xtest)

"""## Log transform training target"""

if logtarget == 1:
    ytrain = np.log1p(ytrain)

#===============================================================================
# callbacks
cb_early = keras.callbacks.EarlyStopping(
    monitor='val_mae', # val_auc for classification
    patience=PARAMS_FIT['patience'],
    verbose=0
)

# cb_checkpt = keras.callbacks.ModelCheckpoint("model_at_epoch_{epoch}.h5")
# cb_lr = lrcurve.KerasLearningCurve()
callbacks = [cb_early]
#===============================================================================

n_feats = len(features)
model = get_keras_model(PARAMS_MODEL,METRICS,n_feats)

history = model.fit(
    Xtrain,
    ytrain,
    batch_size=PARAMS_FIT['batch_size'],
    epochs=PARAMS_FIT['epochs'],
    verbose=0,
    callbacks=callbacks,
    validation_split = PARAMS_FIT['validation_split'],
)

#===============================================================================
# model evaluation
ypreds = model.predict(Xtest).flatten()
if logtarget == 1:
    ypreds = np.expm1(ypreds)

#===============================================================================
# time taken
time_taken = time.time() - time_start_notebook
ncols = len(features)

# if verbose print additional info
if verbose:
    print(model.summary())
    util.print_regr_eval(ytest,ypreds,ncols)
    util.print_time_taken(time_taken)

# write results to a file
if not os.path.exists('outputs'):
    os.makedirs('outputs')

ofile = ('outputs/'   + name  +
        '_scaling_'   + str(scaling) +
        '_logtarget_' + str(logtarget) +
        '.csv')

util.write_regr_eval(ytest,ypreds,ncols,ofile)

# commands
command = """

python model_keras.py --name keras --scaling standard --log 1 --sq 0 \
    --logsq 0 --logtarget 1 --dummy 0 --dummy_cat 0 --verbose 1


python model_keras.py
"""

