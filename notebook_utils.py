"""
This script contains utility functions that are used on the jupyter notebooks to automate functionalities such as
graph representations of the results, model scoring e.t.c.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def mo_reg_scorer(model, X, y):
    """
    A custom scorer to be used for multi-output regression model executions. Calculates both uniform average
    and raw value scores for r2 score, MAE, RMSE & MAPE.

    @param model: the trained model to get scores for.
    @param X: the independent variables to be predicted.
    @param y: the actual labels to compare with model predictions and get scores.
    @return: a dictionary containing the aforementioned scores and the actual predictions.
    """
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
            'ua_score_mape': ua_score_mape, 'rv_scores_mape': rv_scores_mape,
            'predictions': y_pred}


def run_xgb_model(X_train, y_train, X_test=None,  y_test=None):
    # Setup xgb regressor and meta-regressor for multi-output
    xgb_reg = xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.1, max_depth=4, n_estimators=90, n_jobs=-1)
    mo_meta_reg = MultiOutputRegressor(xgb_reg)
    mo_meta_reg.fit(X_train, y_train)

    # If test set is given, predict test set, if not, predict train set.
    if X_test is None or y_test is None:
        scores = mo_reg_scorer(mo_meta_reg, X_train, y_train)
    else:
        scores = mo_reg_scorer(mo_meta_reg, X_test, y_test)

    # Return prediction scores & feature importances
    return {
        'model': "XGBoost",
        'scores': scores,
        'feat_imps': np.array([est.feature_importances_ for est in mo_meta_reg.estimators_])
    }


def run_lin_reg_model(X_train, y_train, X_test=None, y_test=None, scale_feat_imps=True):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # If test set is given, predict test set, if not, predict train set.
    if X_test is None or y_test is None:
        scores = mo_reg_scorer(lr_model, X_train, y_train)
    else:
        scores = mo_reg_scorer(lr_model, X_test, y_test)

    # Get absolute values of model coefficients
    feat_imps = abs(lr_model.coef_)

    # Normalize coefficients (if specified)
    if scale_feat_imps:
        feat_imps = normalize(feat_imps)

    # Return model name, prediction scores & feature importances
    return {
        'model': 'Linear Regression',
        'scores': scores,
        'feat_imps': feat_imps
    }


def run_models(X, y, tt_split_ratio=None, show_err_plots=True, show_featimp_plots=False, show_results=True):

    if tt_split_ratio is None:
        # Run xgboost model & get results
        res_xgb = run_xgb_model(X, y)
        # Run Linear Regression model
        res_lr = run_lin_reg_model(X, y)
    else:
        # Split train test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tt_split_ratio, random_state=96)
        # Run xgboost model & get results
        res_xgb = run_xgb_model(X_train, y_train, X_test, y_test)
        # Run Linear Regression model
        res_lr = run_lin_reg_model(X_train, y_train, X_test, y_test)

    # Print results
    if show_results:
        print_results(res_xgb)
        print_results(res_lr)
    # Plot individual MAPE scores
    if show_err_plots:
        scores_barplot(res_xgb['scores']['rv_scores_mape'], y.columns, title='Raw MAPE scores (XGBoost)')
        scores_barplot(res_lr['scores']['rv_scores_mape'], y.columns, title='Raw MAPE scores (Linear Regression)')
    # Plot feature importances
    if show_featimp_plots:
        plot_feature_imps(res_xgb['feat_imps'], X.columns, y.columns)
        plot_feature_imps(res_lr['feat_imps'], X.columns, y.columns)

    return {'xgb': res_xgb, 'lr': res_lr}


def normalize(feat_imps):
    """
    Normalize feature importances such that they add up to 1.

    @param feat_imps: feature importances list ([[f1,f2,...]]) to be normalized.
    @return: normalized feature importances.
    """
    feat_imps = np.transpose(feat_imps)
    feat_imps_norm = feat_imps * (1 / feat_imps.sum(axis=0))

    return np.transpose(feat_imps_norm)


def scores_barplot(scores, y_cols, title='', figsz=(25, 14), sort=True):
    scores_df = pd.DataFrame(columns=y_cols, data=[scores])
    if sort:
        scores_df = scores_df.sort_values(0, axis=1, inplace=False)

    plt.figure(figsize=figsz)
    plt.title(title)
    sns.set(font_scale=1.2)
    splot = sns.barplot(data=scores_df)
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.3f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9),
                       textcoords='offset points')
    plt.show()

    # return scores_sorted


def plot_1_vs_all_feat_imps(feat_imps, subplt_cols=4, figsz=(18, 25)):
    # Init & configure figure
    fig = plt.figure(figsize=figsz)
    fig.subplots_adjust(hspace=0.1, wspace=0.3)
    # Keep track of number of subplots
    subplt_count = 0
    n_subplots = len(feat_imps)
    # Set number of rows based on specified number of columns
    subplt_rows = int(np.ceil(n_subplots / subplt_cols))
    # Build plots
    sns.set(font_scale=1.5)
    for feat_imp_df in feat_imps:
        # feat_imp_df = feat_imp_df.sort_values(0, axis=0, inplace=False)
        ax = fig.add_subplot(subplt_rows, subplt_cols, subplt_count + 1)
        ax.set_title(feat_imp_df.index[0], fontsize=20)
        sns.barplot(data=feat_imp_df, ax=ax, orient="h")
        subplt_count += 1
    plt.show()


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
    """
    Print results of model ran. Used only for results of functions "run_lin_reg_model", "run_xgb_model"
    and "run_models".

    @param results: a dictionary returned by aforementioned functions holding the model title and the scores.
    """
    # Print model scores (uniform average)
    print(results['model'])
    print('R2 score:', results['scores']['ua_score_r2'])
    print('MAE:', results['scores']['ua_score_mae'])
    print('RMSE:', results['scores']['ua_score_rmse'])
    print('MAPE:', results['scores']['ua_score_mape'])
    print()


def print_avg_scores(scores, model=''):
    """
    A general function for presenting model average scores only if "mo_reg_scorer" is used to get them.

    @param scores: the scores returned from "mo_reg_scorer".
    @param model: the model name (title).
    """
    print(model)
    print('R2 score:', scores['ua_score_r2'])
    print('MAE:', scores['ua_score_mae'])
    print('RMSE:', scores['ua_score_rmse'])
    print('MAPE:', scores['ua_score_mape'])
    print()


def scale_split_labels(data, scaling='std'):
    """
    Scale data and separate independent variables from target variables
    @param scaling: can be 'std', 'min-max' or None for no scaling
    @param data: the dataset
    @return: X:independent vars, y:dependent vars
    """
    if scaling == 'std':
        # Scale dataset
        scaler = StandardScaler()
        dd = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    elif scaling == 'min-max':
        scaler = MinMaxScaler()
        dd = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    else:
        dd = pd.DataFrame(data, columns=data.columns)

    # Split independent variables from target variables
    input_cols = ['slweight(g)']
    X = dd[input_cols]
    y = dd.drop(input_cols, axis=1)

    return X, y


def split_train_test(X, y, test_size=0.2, strat=None):
    """
    Split to train & test sets and then reset indices.

    @param X: train features
    @param y: target variables
    @param test_size: test set size to split
    @param strat: stratification method, (default is "None")
    @return: the now split data
    """
    # TrainTestSplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=96, stratify=strat)
    # Reset indices
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    return X_train, X_test, y_train, y_test



