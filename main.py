import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
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
def mo_reg_scorer(mo_reg_model, X, y, mo='uniform_average'):
    # Get model predictions
    y_pred = mo_reg_model.predict(X)
    # R2
    score_r2 = r2_score(y, y_pred, multioutput=mo)
    # MAE
    score_mae = mean_absolute_error(y, y_pred, multioutput=mo)
    # RMSE
    score_rmse = mean_squared_error(y, y_pred, multioutput=mo, squared=False)
    # MAPE
    score_mape = mean_absolute_percentage_error(y, y_pred, multioutput=mo)

    return {'score_r2': score_r2, 'score_mae': score_mae,
            'score_rmse': score_rmse, 'score_mape': score_mape}


# Fit & evaluate different models
def fit_eval_models(X, y, cv, recs, data_prep='none'):
    # A dictionary to hold cv results (scores) for each algorithm
    # Need to manually add an entry for each model run in the following section
    cv_results = {
        "Linear Regression": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "Decision Tree Regressor": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "Random Forest Regressor": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "Ridge Regressor (multioutput)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "Linear SVR (multioutput)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "Polynomial SVR (multioutput)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "RBF SVR (multioutput)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "XGBoost (multioutput)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "Ridge Regressor (chain)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "Linear SVR (chain)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "Polynomial SVR (chain)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "RBF SVR (chain)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []},
        "XGBoost (chain)": {"avg_scores": [], "raw_scores": [], "avg_train_scores": [], "raw_train_scores": []}

    }

    # 1. Run model for all  cv folds.
    for train_index, test_index in cv.split(X):
        # Get train & test split of current fold.
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

        ################################################################################################################
        # A. Inherently multi-output models
        ################################################################################################################
        # 1. Linear Regression (lr) Model.
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        # Save model's results of current fold
        cv_results['Linear Regression']["avg_scores"].append(mo_reg_scorer(lr_model, X_test, y_test))
        cv_results['Linear Regression']["raw_scores"].append(mo_reg_scorer(lr_model, X_test, y_test, mo='raw_values'))
        cv_results['Linear Regression']["avg_train_scores"].append(mo_reg_scorer(lr_model, X_train, y_train))
        cv_results['Linear Regression']["raw_train_scores"].append(mo_reg_scorer(lr_model, X_train, y_train, mo='raw_values'))
        # 2. Decision Tree Regressor (Inherently multi-output)
        dtr_model = DecisionTreeRegressor()
        dtr_model.fit(X_train, y_train)
        cv_results['Decision Tree Regressor']["avg_scores"].append(mo_reg_scorer(dtr_model, X_test, y_test))
        cv_results['Decision Tree Regressor']["raw_scores"].append(mo_reg_scorer(dtr_model, X_test, y_test, mo='raw_values'))
        cv_results['Decision Tree Regressor']["avg_train_scores"].append(mo_reg_scorer(dtr_model, X_train, y_train))
        cv_results['Decision Tree Regressor']["raw_train_scores"].append(mo_reg_scorer(dtr_model, X_train, y_train, mo='raw_values'))
        # 3. Random Forest Regressor
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        cv_results['Random Forest Regressor']["avg_scores"].append(mo_reg_scorer(rf_model, X_test, y_test))
        cv_results['Random Forest Regressor']["raw_scores"].append(mo_reg_scorer(rf_model, X_test, y_test, mo='raw_values'))
        cv_results['Random Forest Regressor']["avg_train_scores"].append(mo_reg_scorer(rf_model, X_train, y_train))
        cv_results['Random Forest Regressor']["raw_train_scores"].append(mo_reg_scorer(rf_model, X_train, y_train, mo='raw_values'))

        ################################################################################################################
        # B. Multi-output Meta-regressor
        ################################################################################################################
        # 1. Ridge Regressor
        ridge_mo_model = MultiOutputRegressor(Ridge(random_state=96))
        ridge_mo_model.fit(X_train, y_train)
        cv_results['Ridge Regressor (multioutput)']["avg_scores"].append(mo_reg_scorer(ridge_mo_model, X_test, y_test))
        cv_results['Ridge Regressor (multioutput)']["raw_scores"].append(mo_reg_scorer(ridge_mo_model, X_test, y_test, mo='raw_values'))
        cv_results['Ridge Regressor (multioutput)']["avg_train_scores"].append(mo_reg_scorer(ridge_mo_model, X_train, y_train))
        cv_results['Ridge Regressor (multioutput)']["raw_train_scores"].append(mo_reg_scorer(ridge_mo_model, X_train, y_train, mo='raw_values'))
        # 2. Linear SVR
        svr_mo_model = MultiOutputRegressor(LinearSVR())
        svr_mo_model.fit(X_train, y_train)
        cv_results['Linear SVR (multioutput)']["avg_scores"].append(mo_reg_scorer(svr_mo_model, X_test, y_test))
        cv_results['Linear SVR (multioutput)']["raw_scores"].append(mo_reg_scorer(svr_mo_model, X_test, y_test, mo='raw_values'))
        cv_results['Linear SVR (multioutput)']["avg_train_scores"].append(mo_reg_scorer(svr_mo_model, X_train, y_train))
        cv_results['Linear SVR (multioutput)']["raw_train_scores"].append(mo_reg_scorer(svr_mo_model, X_train, y_train, mo='raw_values'))
        # 3. Polynomial SVR (3rd degree)
        svr_poly_mo_model = MultiOutputRegressor(SVR(kernel='poly'))
        svr_poly_mo_model.fit(X_train, y_train)
        cv_results['Polynomial SVR (multioutput)']["avg_scores"].append(mo_reg_scorer(svr_poly_mo_model, X_test, y_test))
        cv_results['Polynomial SVR (multioutput)']["raw_scores"].append(mo_reg_scorer(svr_poly_mo_model, X_test, y_test, mo='raw_values'))
        cv_results['Polynomial SVR (multioutput)']["avg_train_scores"].append(mo_reg_scorer(svr_poly_mo_model, X_train, y_train))
        cv_results['Polynomial SVR (multioutput)']["raw_train_scores"].append(mo_reg_scorer(svr_poly_mo_model, X_train, y_train, mo='raw_values'))
        # 4. RBF SVR
        svr_rbf_mo_model = MultiOutputRegressor(SVR(kernel='rbf'))
        svr_rbf_mo_model.fit(X_train, y_train)
        cv_results['RBF SVR (multioutput)']["avg_scores"].append(mo_reg_scorer(svr_rbf_mo_model, X_test, y_test))
        cv_results['RBF SVR (multioutput)']["raw_scores"].append(mo_reg_scorer(svr_rbf_mo_model, X_test, y_test, mo='raw_values'))
        cv_results['RBF SVR (multioutput)']["avg_train_scores"].append(mo_reg_scorer(svr_rbf_mo_model, X_train, y_train))
        cv_results['RBF SVR (multioutput)']["raw_train_scores"].append(mo_reg_scorer(svr_rbf_mo_model, X_train, y_train, mo='raw_values'))
        # 5. XGBoost
        xgb_mo_model = MultiOutputRegressor(xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.1, max_depth=4, n_estimators=90, n_jobs=-1))
        xgb_mo_model.fit(X_train, y_train)
        cv_results['XGBoost (multioutput)']["avg_scores"].append(mo_reg_scorer(xgb_mo_model, X_test, y_test))
        cv_results['XGBoost (multioutput)']["raw_scores"].append(mo_reg_scorer(xgb_mo_model, X_test, y_test, mo='raw_values'))
        cv_results['XGBoost (multioutput)']["avg_train_scores"].append(mo_reg_scorer(xgb_mo_model, X_train, y_train))
        cv_results['XGBoost (multioutput)']["raw_train_scores"].append(mo_reg_scorer(xgb_mo_model, X_train, y_train, mo='raw_values'))

        ################################################################################################################
        # C. Regressor Chain Meta-regressor
        ################################################################################################################
        # 1. Ridge Regressor
        ridge_chain_model = RegressorChain(Ridge(random_state=96))
        ridge_chain_model.fit(X_train, y_train)
        cv_results['Ridge Regressor (chain)']["avg_scores"].append(mo_reg_scorer(ridge_chain_model, X_test, y_test))
        cv_results['Ridge Regressor (chain)']["raw_scores"].append(mo_reg_scorer(ridge_chain_model, X_test, y_test, mo='raw_values'))
        cv_results['Ridge Regressor (chain)']["avg_train_scores"].append(mo_reg_scorer(ridge_chain_model, X_train, y_train))
        cv_results['Ridge Regressor (chain)']["raw_train_scores"].append(mo_reg_scorer(ridge_chain_model, X_train, y_train, mo='raw_values'))
        # 2. Linear SVR
        svr_chain_model = RegressorChain(LinearSVR())
        svr_chain_model.fit(X_train, y_train)
        cv_results['Linear SVR (chain)']["avg_scores"].append(mo_reg_scorer(svr_chain_model, X_test, y_test))
        cv_results['Linear SVR (chain)']["raw_scores"].append(mo_reg_scorer(svr_chain_model, X_test, y_test, mo='raw_values'))
        cv_results['Linear SVR (chain)']["avg_train_scores"].append(mo_reg_scorer(svr_chain_model, X_train, y_train))
        cv_results['Linear SVR (chain)']["raw_train_scores"].append(mo_reg_scorer(svr_chain_model, X_train, y_train, mo='raw_values'))
        # 3. Polynomial SVR (3rd degree)
        svr_poly_chain_model = RegressorChain(SVR(kernel='poly'))
        svr_poly_chain_model.fit(X_train, y_train)
        cv_results['Polynomial SVR (chain)']["avg_scores"].append(mo_reg_scorer(svr_poly_chain_model, X_test, y_test))
        cv_results['Polynomial SVR (chain)']["raw_scores"].append(mo_reg_scorer(svr_poly_chain_model, X_test, y_test, mo='raw_values'))
        cv_results['Polynomial SVR (chain)']["avg_train_scores"].append(mo_reg_scorer(svr_poly_chain_model, X_train, y_train))
        cv_results['Polynomial SVR (chain)']["raw_train_scores"].append(mo_reg_scorer(svr_poly_chain_model, X_train, y_train, mo='raw_values'))
        # 4. RBF SVR
        svr_rbf_chain_model = RegressorChain(SVR(kernel='rbf'))
        svr_rbf_chain_model.fit(X_train, y_train)
        cv_results['RBF SVR (chain)']["avg_scores"].append(mo_reg_scorer(svr_rbf_chain_model, X_test, y_test))
        cv_results['RBF SVR (chain)']["raw_scores"].append(mo_reg_scorer(svr_rbf_chain_model, X_test, y_test, mo='raw_values'))
        cv_results['RBF SVR (chain)']["avg_train_scores"].append(mo_reg_scorer(svr_rbf_chain_model, X_train, y_train))
        cv_results['RBF SVR (chain)']["raw_train_scores"].append(mo_reg_scorer(svr_rbf_chain_model, X_train, y_train, mo='raw_values'))
        # 5. XGBoost
        xgb_mo_model = RegressorChain(xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.1, max_depth=4, n_estimators=90, n_jobs=-1))
        xgb_mo_model.fit(X_train, y_train)
        cv_results['XGBoost (chain)']["avg_scores"].append(mo_reg_scorer(xgb_mo_model, X_test, y_test))
        cv_results['XGBoost (chain)']["raw_scores"].append(mo_reg_scorer(xgb_mo_model, X_test, y_test, mo='raw_values'))
        cv_results['XGBoost (chain)']["avg_train_scores"].append(mo_reg_scorer(xgb_mo_model, X_train, y_train))
        cv_results['XGBoost (chain)']["raw_train_scores"].append(mo_reg_scorer(xgb_mo_model, X_train, y_train, mo='raw_values'))

    recs.add_records(cv_results, data_prep=data_prep)


# Setup cross-validation generator
n_folds = 5
cv = KFold(n_splits=n_folds, shuffle=True, random_state=96)
# Init Records object to be able to format & save the results
recs = Records()

# A. No preprocessing
# Fit & evaluate models & store results
fit_eval_models(X, y, cv, recs)

# B. Scaling
# Min-Max Scaling
min_max_scaler = MinMaxScaler()
X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)
y_scaled = pd.DataFrame(min_max_scaler.fit_transform(y), columns=y.columns)
fit_eval_models(X_scaled, y_scaled, cv, recs, data_prep='Scaling (Min-Max)')

# Standard Scaling
std_scaler = StandardScaler()
X_scaled = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)
y_scaled = pd.DataFrame(std_scaler.fit_transform(y), columns=y.columns)
fit_eval_models(X_scaled, y_scaled, cv, recs, data_prep='Scaling (Standard)')

# Take a look at all results
print(recs.get_records())

# Export results to csv
recs.export_records_csv("results_3.csv")
