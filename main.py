import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import cross_val_score

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load dataset from csv
data = pd.read_csv("data\\dataset-merged.csv")
# print(data.head())

# Shuffle data
# data = shuffle(data)
# print(data.head())

# Split training samples from labels
input_cols = ['breed', 'sex', 'slaughgr', 'slweight-kg(INPUT)']
X = data[input_cols]
input_cols.append('sheepid')
Y = data.drop(input_cols, axis=1)

# Setup k-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=96)

scores = []
for train_index, test_index in kfold.split(X, Y):
    # print(f"train: {train_index}\ntest: {test_index}")
    # Split train and test sets of current iteration
    X_train, Y_train = X.iloc[train_index], Y.iloc[train_index]
    X_test, Y_test = X.iloc[test_index], Y.iloc[test_index]

    # Setup Multi-output Linear Regressor (molr)
    molr_model = MultiOutputRegressor(Ridge(random_state=96)).fit(X_train, Y_train)
    # pred = molr_model.predict(X_test)
    # Get regressor score.
    r2_score = molr_model.score(X_test, Y_test)
    scores.append(r2_score)
    print("Score (R2):", r2_score)

print("Average Score (R2):", sum(scores)/kfold.get_n_splits())
print("Score variance (R2):", np.var(scores))




