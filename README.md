# King County House Price Prediction Project
In this project I did a detail study of how to get a best model to fit the regression problem of finding the
price of house in King County (where Seattle is located) using various features.
- The modelling notebooks and eda notebooks are located in directory `notebooks`.
- The original data is given in `data/raw/kc_house_data.csv`.
- The reports of EDA are given in html files in path `reports/html/`.

# Findings
I tried following models
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Decision Tree Regressor (CAT methods)
- k-nearest neighbors Regressor
- Random Forest Regressor

> Polynomial Regression with degree 2 gives result of 


Final score is Adjusted R-squared value for test (20% random split) is 0.89 for Random Forest after hyper parameter tuning.
