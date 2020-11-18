# Imports
import numpy as np
import pandas as pd

# special import
from explainx import explainx

# local imports
import config
import util
from util import clean_data
from util import print_regr_eval

# modelling
import catboost as cb


# Parameters
data_path_train = config.data_path_train
data_path_test = config.data_path_test
target = config.target
logtarget = True

params_data = dict(log=True, sq=True, logsq=False, dummy=True, dummy_cat=False)
params_cb = config.params_cb
model_dump_cb = config.model_dump_cb


def main():
    """Main function """
    # Load the data
    df_train = pd.read_csv(data_path_train)
    df_test = pd.read_csv(data_path_test)

    # Data Cleaning and feature selection
    df_train = clean_data(df_train, **params_data)
    df_test = clean_data(df_test, **params_data)
    features = list(sorted(df_train.columns.drop(target)))
    features = [i for i in features if i in df_test.columns]

    df_Xtrain = df_train[features]
    df_Xtest  = df_test[features]
    ytrain = np.array(df_train[target]).flatten()
    ytest  = np.array(df_test[target]).flatten()
    if logtarget:
        ytrain = np.log1p(ytrain)
        ytest = np.log1p(ytest)

    # modelling
    model = cb.CatBoostRegressor()
    model = model.load_model(model_dump_cb)

    ypreds = model.predict(df_test[features])
    ypreds = np.array(ypreds).flatten()
    if logtarget:
        ypreds = np.expm1(ypreds)

    explainx.ai(df_Xtest, ytest, model, model_name="catboost")


if __name__ == "__main__":
    main()
