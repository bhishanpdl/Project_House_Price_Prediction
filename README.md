
<h1 style="background-color:tomato;">Project Description</h1>

In this project, the dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015. There are 19 house features and one dependent feature `price`. The aim of the project is to estimate the house price.


<h1 style="background-color:tomato;">Data processing</h1>

- Linear models and svm benefits from scaling (and removing outliers), I did normalizing and robust scaling.
- Created a new feature `age_after_renovation` using `yr_sales` and `yr_renovated` features.
- `zipcode` has too many unique values, reduced it to 70 values.
- Created a new feature called `zipcode_houses` which gives number of houses in each zipcode.
- Created binned features from `age` and `age_after_renovation`.
- Did `log1p` transformation of continuous numerical features.


<h1 style="background-color:tomato;">Best Results</h1>
After comprehensive data cleaning and variable encodings,
I tried various scikit learn algorithms including stacking and blending. I had created many categorical features and the catboost algorithm after standard scaling gave me the best adjusted r-squared value.

![](images/boost_res.png)

<h1 style="background-color:tomato;">Big data modelling</h1>

- `scikit-learn` and `pandas` can not deal with large data (`>1GB`). To scale up the project, I used big data platform `PySpark`.
- `spark` is a scala package and `pyspark` is the a python wrapper around it.
- In `pyspark`, `mllib` is deprecated, so, I used only `pyspark.ml`.
- I used `Random Forest` in pyspark and tuned the hyper parameters to get the best Adjusted R-squared value.


<h1 style="background-color:tomato;">Some of the EDA results</h1>

![](images/correlation_matrix.png)
![](images/correlation_matrix2.png)
![](images/sns_heatmap.png)
![](images/some_histograms.png)
![](images/bedroom_bathrooms_waterfron_view.png)
![](images/bedroom_counts.png)



<h1 style="background-color:tomato;">Project Notebooks</h1>

|  Notebook | Rendered   | Description  |  Author |
|---|---|---|---|
| a00_best_model2.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a00_best_model2.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a00_best_model2.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| a00_best_model_xgb.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a00_best_model_xgb.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a00_best_model_xgb.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| a01_data_proc.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a01_data_proc.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a01_data_proc.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| a02_data_proc_script.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a02_data_proc_script.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a02_data_proc_script.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| a03_stats.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a03_stats.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/a03_stats.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b01_eda.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b01_eda.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b01_eda.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b02_eda_bokeh.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b02_eda_bokeh.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b02_eda_bokeh.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b03_eda_plotly.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b03_eda_plotly.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b03_eda_plotly.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b04_eda_pixiedust.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b04_eda_pixiedust.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b04_eda_pixiedust.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b05_eda_pandas_profiling.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b05_eda_pandas_profiling.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/b05_eda_pandas_profiling.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c01_sklearn_linear_and_polynomial.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c01_sklearn_linear_and_polynomial.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c01_sklearn_linear_and_polynomial.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c02_statsmodels_ols.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c02_statsmodels_ols.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c02_statsmodels_ols.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c03_sklearn_methods.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c03_sklearn_methods.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c03_sklearn_methods.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c04_rf.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c04_rf.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c04_rf.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c05_rf_feat_imp.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c05_rf_feat_imp.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c05_rf_feat_imp.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c06_sklearn_methods.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c06_sklearn_methods.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c06_sklearn_methods.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c07_sklearn_select_kbest.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c07_sklearn_select_kbest.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c07_sklearn_select_kbest.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c08_boosting_gbr.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c08_boosting_gbr.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c08_boosting_gbr.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c09_boosting_xgb.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c09_boosting_xgb.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c09_boosting_xgb.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c10_boosting_lgb.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c10_boosting_lgb.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c10_boosting_lgb.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c11_boosting_catboost.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c11_boosting_catboost.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c11_boosting_catboost.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c12_ensemble_stacking_blending.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c12_ensemble_stacking_blending.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c12_ensemble_stacking_blending.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c13_pycaret.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c13_pycaret.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/c13_pycaret.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| d01_bigdata_pyspark.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/d01_bigdata_pyspark.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/d01_bigdata_pyspark.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| d02_bigdata_pyspark_rf_tuning.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/d02_bigdata_pyspark_rf_tuning.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/d02_bigdata_pyspark_rf_tuning.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| e01_model_interpretation_rf.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/e01_model_interpretation_rf.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_House_Price_Prediction/blob/master/notebooks/e01_model_interpretation_rf.ipynb)  |   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
