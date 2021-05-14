import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

# Load dataset from csv
data = pd.read_csv("data\\dataset-merged.csv")

# Drop unwanted columns
data.drop(['sheepid'], axis=1, inplace=True)
data.drop(['slaughgr'], axis=1, inplace=True)
data.drop(['sex'], axis=1, inplace=True)
data.drop(['breed'], axis=1, inplace=True)
# print(data.head())

# Scale dataset
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split independent variables from target variables
input_cols = ['slweight(g)']
X = data_scaled[input_cols]
y = data_scaled.drop(input_cols, axis=1)
# print(X.head())
# print(y.head())

xgb_model = MultiOutputRegressor(xgb.XGBRegressor())
params_dist = {'n_estimators': stats.randint(70, 150),
               'learning_rate': stats.uniform(0.01, 0.1),
               'subsample': stats.uniform(0.3, 0.7),
               'max_depth': [3, 4, 5, 6],
               'colsample_bytree': stats.uniform(0.5, 0.45),
               'min_child_weight': [1, 2, 3]}

reg = RandomizedSearchCV(xgb.XGBRegressor(), param_distributions=params_dist, n_iter=10,
                         scoring='neg_mean_absolute_percentage_error', error_score=0, verbose=3, n_jobs=-1)

num_folds = 5
cv = KFold(n_splits=num_folds, shuffle=True, random_state=96)

estimators = []
results = np.zeros(len(X))
score = 0
for train_index, test_index in cv.split(X):
    # Get train & test split of current fold.
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
    reg.fit(X_train, y_train)

    estimators.append(reg.best_estimator_)
    results[test_index] = reg.predict(X_test)
    score += mean_absolute_percentage_error(y_test, results[test_index])
score /= num_folds
print(score)
