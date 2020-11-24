#!/usr/bin/env python
import pandas as pd
import pandas_profiling

def get_report_pandas_profiling(ifile,ofile):
    # Data
    df = pd.read_csv(ifile)
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(ofile)

if __name__ == '__main__':
    ifile = '../data/raw/kc_house_data.csv'
    ofile = '../reports/report_pandas_profiling.html'
    get_sweetviz_report(ifile,ofile)
