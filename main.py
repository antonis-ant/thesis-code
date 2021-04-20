import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Records import Records

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load dataset from csv
data = pd.read_csv("data\\dataset-merged.csv")
# print(data.head())

# Split training samples from labels
input_cols = ['breed', 'sex', 'slaughgr', 'slweight(g)']
X = data[input_cols]
input_cols.append('sheepid')
y = data.drop(input_cols, axis=1)


# Define a custom scorer for multi-output regression
def mo_reg_scorer(mo_reg_model, X, y):
    # Get model predictions
    y_pred = mo_reg_model.predict(X)
    # R2
    score_r2 = r2_score(y, y_pred, multioutput='uniform_average')
    scores_r2 = r2_score(y, y_pred, multioutput='raw_values')
    # MAE
    score_mae = mean_absolute_error(y, y_pred, multioutput='uniform_average')
    scores_mae = mean_absolute_error(y, y_pred, multioutput='raw_values')
    # RMSE
    score_rmse = mean_squared_error(y, y_pred, multioutput='uniform_average', squared=False)
    scores_rmse = mean_squared_error(y, y_pred, multioutput='raw_values', squared=False)
    # MAPE
    score_mape = mean_absolute_percentage_error(y, y_pred, multioutput='uniform_average')
    scores_mape = mean_absolute_percentage_error(y, y_pred, multioutput='raw_values')

    return {
        'avg_scores': {'score_r2': score_r2, 'score_mae': score_mae,
                       'score_rmse': score_rmse, 'score_mape': score_mape},
        'raw_scores': {'score_r2': scores_r2, 'score_mae': scores_mae,
                       'score_rmse': scores_rmse, 'score_mape': scores_mape}}


# Fit & evaluate different models
def fit_eval_models(X, y, regressors, cv, recs, data_prep='none'):
    # Initialize dictionary to hold cv results
    cv_results = dict()
    for key in regressors.keys():
        cv_results[key] = {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []}

    # Run models for all cv folds.
    for train_index, test_index in cv.split(X):
        # Get train & test split of current fold.
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

        # Run & evaluate each regressor model for current cv fold.
        for key in regressors:
            # Get regressor model
            model = regressors[key]
            model.fit(X_train, y_train)
            # Get prediction scores for test & train sets
            scores = mo_reg_scorer(model, X_test, y_test)
            scores_train = mo_reg_scorer(model, X_train, y_train)
            # Save model's results of current fold
            cv_results[key]["avg_scores"].append(scores['avg_scores'])
            cv_results[key]["raw_scores"].append(scores['raw_scores'])
            cv_results[key]["avg_train_scores"].append(scores_train['avg_scores'])
            cv_results[key]["raw_train_scores"].append(scores_train['raw_scores'])

    # Add results to records to organize them and average the cv scores.
    recs.add_records(cv_results, data_prep=data_prep)


# Define the regressor models we want to try
regressors = {
    # Inherently multi-output
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    # Using multi-output meta-regressor
    'Ridge Regressor (multioutput)': MultiOutputRegressor(Ridge(random_state=96)),
    'Linear SVR (multioutput)': MultiOutputRegressor(LinearSVR()),
    'Polynomial SVR (multioutput)': MultiOutputRegressor(SVR(kernel='poly')),
    'RBF SVR (multioutput)': MultiOutputRegressor(SVR(kernel='rbf')),
    'XGBoost (multioutput)': MultiOutputRegressor(xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.1,
                                                                   max_depth=4, n_estimators=90, n_jobs=-1)),
    # Using regressor chain
    'Ridge Regressor (chain)': RegressorChain(Ridge(random_state=96)),
    'Linear SVR (chain)': RegressorChain(LinearSVR()),
    'Polynomial SVR (chain)': RegressorChain(SVR(kernel='poly')),
    'RBF SVR (chain)': RegressorChain(SVR(kernel='rbf')),
    'XGBoost (chain)': RegressorChain(xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.1,
                                                       max_depth=4, n_estimators=90, n_jobs=-1))
}
# Setup cross-validation generator
n_folds = 5
cv = KFold(n_splits=n_folds, shuffle=True, random_state=96)
# Init Records object to be able to format & save the results
recs = Records()

# A. No preprocessing
# Fit & evaluate models & store results
fit_eval_models(X, y, regressors, cv, recs)

# B. Scaling
# Min-Max Scaling
min_max_scaler = MinMaxScaler()
X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)
y_scaled = pd.DataFrame(min_max_scaler.fit_transform(y), columns=y.columns)
fit_eval_models(X_scaled, y_scaled, regressors, cv, recs, data_prep='Scaling (Min-Max)')
# Standard Scaling
std_scaler = StandardScaler()
X_scaled = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)
y_scaled = pd.DataFrame(std_scaler.fit_transform(y), columns=y.columns)
fit_eval_models(X_scaled, y_scaled, regressors, cv, recs, data_prep='Scaling (Standard)')

# Take a look at all results
print(recs.get_records())

# Export results to csv
# recs.export_records_csv("results_4.csv")
