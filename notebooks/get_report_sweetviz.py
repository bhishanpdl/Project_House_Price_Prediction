#!/usr/bin/env python
import pandas as pd
import sweetviz

def get_report_sweetviz(ifile,ofile):
    # config
    sweetviz.config_parser.read_string("[Layout]\nshow_logo=0")

    # Data
    df = pd.read_csv(ifile)
    print(f'shape: {df.shape}')

    my_report = sweetviz.analyze([df,'Full data'])
    my_report.show_html(ofile)

if __name__ == '__main__':
    ifile = '../data/raw/kc_house_data.csv'
    ofile = '../reports/report_sweetviz.html'
    get_report_sweetvis(ifile,ofile)
