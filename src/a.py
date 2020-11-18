#!python
# -*- coding: utf-8 -*-#
"""
* File Name : a.py

* Purpose :

* Creation Date : Nov 17, 2020 Tue

* Last Modified : Tue Nov 17 21:28:26 2020

* Created By :  Bhishan Poudel

"""
# Imports
path_report_pandas_profiling = '../reports/report_pandas_profiling.html'
with open(path_report_pandas_profiling,'r',encoding='utf-8') as fi:
        html_file = fi.readlines()
        html_file = '\n'.join(html_file)

print(html_file)
