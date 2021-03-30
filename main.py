import pandas as pd

from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor

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
def mo_reg_scorer(mo_reg_model, X, y, mo='uniform_average'):
    # Get model predictions
    y_pred = mo_reg_model.predict(X)
    # R2
    score_r2 = r2_score(y, y_pred, multioutput=mo)
    # MAE
    score_mae = mean_absolute_error(y, y_pred, multioutput=mo)
    # MSE
    score_rmse = mean_squared_error(y, y_pred, multioutput=mo, squared=False)
    # MAPE
    score_mape = mean_absolute_percentage_error(y, y_pred, multioutput=mo)

    return {'score_r2': score_r2, 'score_mae': score_mae,
            'score_rmse': score_rmse, 'score_mape': score_mape}


# Fit & evaluate different models
def fit_eval_models(X, y, cv, recs, data_prep='none'):
    # 1. Linear Regression (Inherently multi-output)
    # Initialize model
    lr_model = LinearRegression()
    # Fit & predict using cross validation
    cv_results = cross_validate(lr_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 2. Decision Tree Regression (Inherently multi-output)
    dtr_model = DecisionTreeRegressor()
    # Fit & predict using cross validation
    cv_results = cross_validate(dtr_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 3. Random Forest Regression (Inherently multi-output)
    rf_model = RandomForestRegressor()
    # Fit & predict using cross validation
    cv_results = cross_validate(rf_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 4. Ridge Regression with MultiOutputRegressor (Meta-regressor)
    # Initialize model
    ridge_mo_model = MultiOutputRegressor(Ridge(random_state=96))
    # Fit & predict using cross validation
    cv_results = cross_validate(ridge_mo_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 5. Linear SVR with MultiOutputRegressor (Meta-regressor)
    # Initialize model
    svr_mo_model = MultiOutputRegressor(LinearSVR(max_iter=10000))
    # Fit & predict using cross validation
    cv_results = cross_validate(svr_mo_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 6. Polynomial SVR (3rd degree) with MultioutputRegressor (Meta-regressor)
    svr_poly_mo_model = MultiOutputRegressor(SVR(kernel='poly'))
    cv_results = cross_validate(svr_poly_mo_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 7. Polynomial SVR (3rd degree) with MultioutputRegressor (Meta-regressor)
    svr_rbf_mo_model = MultiOutputRegressor(SVR(kernel='rbf'))
    cv_results = cross_validate(svr_rbf_mo_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 8. Ridge Regression with RegressorChain (Meta-regressor)
    # Initialize model
    ridge_chain_model = RegressorChain(Ridge(random_state=96), order='random')
    # Fit & predict using cross validation
    cv_results = cross_validate(ridge_chain_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 9. Linear SVR Regression with RegressorChain (Meta-regressor)
    # Initialize model
    ridge_chain_SVR_model = RegressorChain(LinearSVR(max_iter=10000), order='random')
    # Fit & predict using cross validation
    cv_results = cross_validate(ridge_chain_SVR_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 10. Polynomial (3rd degree) SVR with ChainRegressor (Meta-regressor)
    svr_poly_chain_model = RegressorChain(SVR(kernel='poly'))
    cv_results = cross_validate(svr_poly_chain_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())

    # 11. rbf SVR with ChainRegressor (Meta-regressor)
    svr_rbf_chain_model = RegressorChain(SVR(kernel='rbf'))
    cv_results = cross_validate(svr_rbf_chain_model, X, y, cv=cv, scoring=mo_reg_scorer, return_train_score=True,
                                return_estimator=True)
    # Save results
    recs.add_record(cv=cv, cv_results=cv_results, data_prep=data_prep)
    # print(recs.get_last_rec())


# Setup cross-validation generator
cv_k = 5
cv = KFold(n_splits=cv_k, shuffle=True, random_state=96)
# Init Records object to be able to format & save the results
recs = Records()

# A. No preprocessing
# Fit & evaluate models & store results
fit_eval_models(X, y, cv, recs)

# B. Scaling
# Min-Max Scaling
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
y_scaled = min_max_scaler.fit_transform(y)
fit_eval_models(X_scaled, y_scaled, cv, recs, data_prep='Scaling (Min-Max)')

# Standard Scaling
std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X)
y_scaled = std_scaler.fit_transform(y)
fit_eval_models(X_scaled, y_scaled, cv, recs, data_prep='Scaling (Standard)')

# Take a look at all results
print(recs.get_records(compact=False))

# Export results to csv
recs.export_records_csv("results_2.csv")
