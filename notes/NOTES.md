1. In regression, when I used the k-fold cross validation, without normalizing
   the features with large values (price, sqft_living, sqft_living15, etc.),
   the linear regression models failed miserably and gave ridiculous results.
   So, First log tranform the numerical features, then standard scale and 
    only fit the model.