import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn import metrics
import config

# params
SEED = config.SEED

# deep learning
import keras

def set_random_seed(seed):
    import os
    import random
    import numpy as np
    import tensorflow as tf

    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_keras_model(params,metrics,n_feats):
    # num of layers
    n_layers = len([i for i in list(params.keys()) if i.endswith('_units')])

    # layers
    model = keras.Sequential(name='Sequential')

    # layer 1
    model.add(keras.layers.Dense(
        params['L1_units'],
        activation=params['L1_act'],
        kernel_initializer=params['L1_kernel_init'],
        kernel_regularizer=params['L1_kernel_reg'],
        bias_regularizer=params['L1_bias_reg'],
        input_shape=(n_feats,),
        name='Layer_1'
        ))
    model.add(keras.layers.Dropout(params['L1_dropout'],
                                seed=SEED,
                                name='Dropout_1'))

    # middle layers
    for i in range(2,n_layers+1): # 2,3, etc
        model.add(keras.layers.Dense(
            params[f'L{i}_units'],
            activation=params[f'L{i}_act'],
            kernel_initializer=params[f'L{i}_kernel_init'],
            kernel_regularizer=params[f'L{i}_kernel_reg'],
            bias_regularizer=params[f'L{i}_bias_reg'],
            name=f'Layer_{i}'),)
        model.add(keras.layers.Dropout(
            params[f'L{i}_dropout'],
            seed=SEED,
            name=f"Dropout_{i}"))

    # last layer is dense 1 with activation sigmoid
    model.add(keras.layers.Dense(
        1,
        activation=None, # activation = None or linear does nothing
        name=f'Layer_{n_layers+1}'
        ))

    #=================================================== compile
    model.compile(
        optimizer=params['optimizer'],
        loss='mse',
        metrics=metrics
        )

    return model

def plot_keras_history(h, metric,figsize=(12,8),ofile=None):
    """Plot training vs validation plots for metric and loss.
    For example: metric = mae

    """
    # history
    if not isinstance(h,dict):
        h = h.history

    # prepare plot
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True,figsize=figsize)

    # metric
    plt.subplot(211)
    plt.plot(h[metric])
    plt.plot(h['val_'+metric])
    plt.title('Training vs Validation '+ metric.upper())
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # save
    plt.tight_layout()
    if ofile:
        plt.savefig(ofile,dpi=300)

    # show plot
    plt.draw()
    plt.show()