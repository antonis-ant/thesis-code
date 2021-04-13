import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor


# Define Custom scorer that calculates uniform average and raw values scores.
def mo_reg_scorer(model, X, y):
    y_pred = model.predict(X)

    # Calculate both Uniform average (ua) & raw values (rv) scores/errors
    # R2
    ua_score_r2 = r2_score(y, y_pred, multioutput='uniform_average')
    rv_scores_r2 = r2_score(y, y_pred, multioutput='raw_values')
    # MAE
    ua_score_mae = mean_absolute_error(y, y_pred, multioutput='uniform_average')
    rv_scores_mae = mean_absolute_error(y, y_pred, multioutput='raw_values')
    # RMSE
    ua_score_rmse = mean_squared_error(y, y_pred, multioutput='uniform_average', squared=False)
    rv_scores_rmse = mean_squared_error(y, y_pred, multioutput='raw_values', squared=False)
    # MAPE
    ua_score_mape = mean_absolute_percentage_error(y, y_pred, multioutput='uniform_average')
    rv_scores_mape = mean_absolute_percentage_error(y, y_pred, multioutput='raw_values')

    return {'ua_score_r2': ua_score_r2, 'rv_scores_r2': rv_scores_r2,
            'ua_score_mae': ua_score_mae, 'rv_scores_mae': rv_scores_mae,
            'ua_score_rmse': ua_score_rmse, 'rv_scores_rmse': rv_scores_rmse,
            'ua_score_mape': ua_score_mape, 'rv_scores_mape': rv_scores_mape}


def run_xgb_model(X_train, X_test, y_train, y_test):
    # Setup xgb regressor and meta-regressor for multi-output
    xgb_reg = xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.1, max_depth=4, n_estimators=90, n_jobs=-1)
    mo_meta_reg = MultiOutputRegressor(xgb_reg)
    # Fit
    mo_meta_reg.fit(X_train, y_train)

    # Return prediction scores & feature importances
    return {
        'model': "XGBoost",
        'scores': mo_reg_scorer(mo_meta_reg, X_test, y_test),
        'feat_imps': np.array([est.feature_importances_ for est in mo_meta_reg.estimators_])
    }


def run_lin_reg_model(X_train, X_test, y_train, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    return {
        'model': 'Linear Regression',
        'scores': mo_reg_scorer(lr_model, X_test, y_test),
        'feat_imps': lr_model.coef_
    }


def scores_barplot(scores, y_cols, title='', figsz=(25, 14)):
    scores_df = pd.DataFrame(columns=y_cols, data=[scores])

    plt.figure(figsize=figsz)
    plt.title(title)
    sns.barplot(data=scores_df)


def plot_feature_imps(feat_imps, X_colnames, y_colnames, subplt_cols=4, figsz=(18, 25)):
    # Init & configure figure
    fig = plt.figure(figsize=figsz)
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    # fig.suptitle('Feature Importances for Each Target Variable', fontsize=10)
    # keep track of number of subplots
    subplt_count = 0
    n_subplots = len(feat_imps)
    # number of rows based on specified number of columns
    subplt_rows = int(np.ceil(n_subplots/subplt_cols))
    # Build plots
    for fi in feat_imps:
        df = pd.DataFrame(data=[fi], columns=X_colnames)
        ax = fig.add_subplot(subplt_rows, subplt_cols, subplt_count + 1)
        ax.set_title(y_colnames[subplt_count])
        sns.barplot(data=df, ax=ax)
        subplt_count += 1


def print_results(results):
    # Print model scores (uniform average)
    print(results['model'])
    print('R2 score:', results['scores']['ua_score_r2'])
    print('MAE:', results['scores']['ua_score_mae'])
    print('RMSE:', results['scores']['ua_score_rmse'])
    print('MAPE:', results['scores']['ua_score_mape'])
    print()


def run_models(X, y, show_plots=True, show_results=True):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=96)

    # Run xgboost model & get results
    res_xgb = run_xgb_model(X_train, X_test, y_train, y_test)
    # Run Linear Regression model
    res_lr = run_lin_reg_model(X_train, X_test, y_train, y_test)

    # Print results
    if show_results:
        print_results(res_xgb)
        print_results(res_lr)

    # Plot individual RMSE scores
    if show_plots:
        scores_barplot(res_xgb['scores']['rv_scores_rmse'], y.columns, title='Raw RMSE scores (XGBoost)')
        scores_barplot(res_lr['scores']['rv_scores_rmse'], y.columns, title='Raw RMSE scores (Linear Regression)')

    return {'xgb': res_xgb, 'lr': res_lr}


# def try_permutations(X, y):
#     df_cols = ['Output Feature Used',
#                'XGBoost RMSE',
#                'Linear Regression RMSE',
#                'XGBoost R2',
#                'Linear Regression R2']
#     res_df = pd.DataFrame(columns=df_cols)
#
#     for col in y.columns:
#         X_alt = X
#         X_alt[col] = y[[col]]
#         y_alt = y.drop([col], axis=1)
#
#         print('X_alt: ', X_alt.columns)
#         print('y_alt: ', y_alt.columns)
#
#         r = run_models(X_alt, y_alt, show_plots=False, show_results=False)
#
#         res_df = res_df.append({
#             'Output Feature Used': col,
#             'XGBoost RMSE': r['xgb']['scores']['ua_score_rmse'],
#             'Linear Regression RMSE': r['lr']['scores']['ua_score_rmse'],
#             'XGBoost R2': r['xgb']['scores']['ua_score_r2'],
#             'Linear Regression R2': r['lr']['scores']['ua_score_r2']
#         }, ignore_index=True)
#
#     print(res_df)
