# Project: King County House Price Prediction

# File Structure
- Raw input data is at `data/raw/kc_house_data.csv`
- `notebooks` directory has all the notebooks for exploratory data analysis,
   visualization and modelling and model interpretation.
- The project is divided into multiple parts:
  + Data processing `notebooks/regression_data_processing.ipynb`
  + Data visualization `notebooks/regression_data_visualization.ipynb`
  + Data profiling `notebooks/regression_eda_with_pandas_profiling.ipynb`
  + Simplest model: linear regression `notebooks/regression_modelling_linear_and_polynomial_sklearn.ipynb`
  + Regression using statsmodels `notebooks/regression_modelling_linear_ols_statsmodels.ipynb`
  + Finding best Adjusted R-squared model: `notebooks/regression_modelling_sklearn_best_r2.ipynb`
  + I found Random Forest gives the best result for this data: `notebooks/regression_modelling_random_forest.ipynb`


# Data Description

This dataset contains house sale prices for King County,
which includes Seattle.
It includes homes sold between May 2014 and May 2015.

- Dependent features: 1 (price)
- Features : 19 home features
- Id:  House ID

Task: Try to estimate the price of house based on given features.
![](../data/raw/data_description.png)

# Data Cleansing and Feature Engineering
The data cleaning notebook is `notebooks/a01_regression_data_processing.ipynb`
and `notebooks/a01_regression_data_processing_script.ipynb`. The script is located in
`src/data/data_cleaning.py`.
For the Random Forest model creating new featrues and doing log tranforming or
creating dummy varibles may not be that helpful, but they are incredibly useufl
in case of linear regression and other non-tree based algorithms.

I created age and age_after_renovation from year columns. Also create dummies for
all the categorical featrues and created log transformed varibales for features with
large values like (sqft_living, sqft_living15, sqft_lot, etc.)

I also created new features zipcode_houses that gives number of houses present in
given zipcode. Also, zipcode categorical feature has 70 different classes,
creating 70 dummies variables may not be a good idea, so I created 10 dummies from
it based on most expensive houses zipcodes. I could have also done most frequent top 10 zipcodes or any other logics but I liked the idea of most expensive houses.

# Data visualization and EDA
I used `matplotlib`, `seaborn`, `pandas` and `plotly` for the data visualization
of various features.