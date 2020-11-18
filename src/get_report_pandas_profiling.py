# Imports
import numpy as np
import pandas as pd
import pandas_profiling

# local imports
import config

# Parameters
data_path_train = config.data_path_train
path_report_pandas_profiling = config.path_report_pandas_profiling

def main():
    df_train = pd.read_csv(data_path_train)

    profile = pandas_profiling.ProfileReport(df_train)

    # write output html
    profile.to_file(path_report_pandas_profiling)

if __name__ == '__main__':
    # run the program
    main()