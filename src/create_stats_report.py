# Load the libraries
import numpy as np
import pandas as pd

import sweetviz
sweetviz.config_parser.read_string("[Layout]\nshow_logo=0")

# Data

df = pd.read_csv('../data/raw/kc_house_data.csv')
print(f'shape: {df.shape}')

my_report = sweetviz.analyze([df,'Full data'])
my_report.show_html('../reports/stats_report.html')
