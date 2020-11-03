import os

# data
dat_dir = os.path.join('..','data')
data_path = os.path.join(dat_dir, 'raw/kc_house_data.csv')
data_path_raw = os.path.join(dat_dir, 'raw/kc_house_data.csv')
data_path_clean = os.path.join(dat_dir, 'processed/data_cleaned_encoded.csv')
compression = None

# params
target = 'price'
train_size = 0.8
test_size = 1-train_size
SEED = 100

# data Processing
cols_log_small = ['price','sqft_living','sqft_living15',
            'sqft_lot','sqft_lot15']
cols_drop_small = ['id','date']

cols_sq = ['bedrooms','bathrooms','floors','waterfront','view',
    'age','age_after_renovation','log1p_sqft_living','log1p_sqft_lot',
    'log1p_sqft_above','log1p_sqft_basement',
    'log1p_sqft_living15','log1p_sqft_lot15']
cols_drop = ['id', 'date', 'zipcode_top10']

# params
# NOTE: for small model with only raw features, params_rf_small gives better rmse
params_rf_small = dict(n_estimators= 155,random_state=SEED,
    max_features='auto',max_depth=23,min_samples_split=10)

params_rf = dict(n_estimators=1200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=5,
                max_features=None,
                oob_score=True,
                n_jobs=-1,
                random_state=SEED)

params_xgb = dict(n_jobs=-1,
                random_state=SEED,
                objective='reg:squarederror',
                n_estimators=1200,
                max_depth=3,
                reg_alpha=1,
                reg_lambda=5,
                subsample=1,
                gamma=0,
                min_child_weight=1,
                colsample_bytree=1,
                learning_rate=0.1)

params_lgb = {
    'boosting_type': 'gbdt',
    'colsample_by_tree': 0.67211,
    'learning_rate': 0.02169,
    'max_depth': 15,
    'n_estimators': 750,
    'num_leaves': 38,
    'reg_lambda': 0.604
            }

params_cb ={'depth': 7,
'early_stopping_rounds': 200,
'eval_metric': 'RMSE',
'iterations':2503,
'l2_leaf_reg': 3,
'learning_rate': 0.03,
'loss_function': 'RMSE',
'random_state': 123,
'subsample': 0.8,
'verbose': False}

lst_cat_features = ['bedrooms','waterfront','view','condition','grade','zipcode']
params_cb['cat_features'] = lst_cat_features