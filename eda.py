import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load dataset from csv
data = pd.read_csv("data\\dataset-merged.csv")
# print(data.head().transpose())

# Split training samples from labels
input_cols = ['breed', 'sex', 'slaughgr', 'slweight(g)']
X = data[input_cols]
input_cols.append('sheepid')
y = data.drop(input_cols, axis=1)

# Split train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=96)

# Setup xgb regressor and meta-regressor for multi-output
xgb_reg = xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.1, max_depth=4, n_estimators=90, n_jobs=-1)
mo_meta_reg = MultiOutputRegressor(xgb_reg)
# Fit & predict
mo_meta_reg.fit(X_train, y_train)
y_pred = mo_meta_reg.predict(X_test)
# Evaluate Test
score_rmse = mean_squared_error(y_test, y_pred, multioutput='raw_values', squared=False)
score_r2 = r2_score(y_test, y_pred, multioutput='raw_values')
# Evaluate Train
# y_train_pred = mo_meta_reg.predict(X_train)
# train_score_rmse = mean_squared_error(y_train, y_train_pred, multioutput='uniform_average', squared=False)
# train_score_r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')

print("XGBoost")
print("RMSE: ", score_rmse)
print("R2: ", score_r2)
# print("RMSE (train): ", train_score_rmse)
# print("R2 (train): ", train_score_r2)

# Get feature importances
for est in mo_meta_reg.estimators_:
    print(est.feature_importances_)

# print(mo_meta_reg.estimators_[0].feature_importances_)



# # Check correlation of input variables
# # Separate the input variables
# data_inputs = data[input_cols]
# # Get their correlation matrix
# input_cols_corr = data_inputs.corr()
# sns.heatmap(input_cols_corr, annot=False, cmap='Reds')
# plt.show()




