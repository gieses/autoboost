import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost
from scipy.stats import pearsonr
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import make_scorer, mean_squared_error
import lightgbm
from autoboost import optimizer

warnings.simplefilter(action='ignore', category=FutureWarning)


def prepare_diamonds():
    df = sns.load_dataset("diamonds")
    cat_columns = ["cut", "color", "clarity"]

    label_encoder = preprocessing.LabelEncoder()
    for column in cat_columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df


# get the diamonds dataframe with xgboost, categorical variable handling
df = prepare_diamonds()
X = df.filter(regex="^(?!price)")
y = df.filter(regex="price")

X = X.sample(15000, random_state=15)
y = y.loc[X.index]
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

nfolds = 3
verbose = -1


x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=1)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# default
xgb_default = xgboost.XGBRegressor()
xgb_default.fit(x_train, y_train)
y_test_hat = xgb_default.predict(x_test)
print(f"xgboost: {mean_squared_error(y_test, y_test_hat):.2f}")
print(f"xgboost: {pearsonr(np.ravel(y_test.values), y_test_hat)[0]:.2f}")

# autoboost
bo = optimizer.BoostingOptimizer(initial_model=xgboost.XGBRegressor(), scorer=mse_scorer, n_folds=nfolds,
                                 verbose=verbose)
clf = bo.fit(x_train, y_train)
y_test_hat = clf.best_estimator_.predict(x_test)
print(f"xgboost autoboost: {mean_squared_error(y_test, y_test_hat):.2f}")
print(f"xgboost autoboost: {pearsonr(np.ravel(y_test.values), y_test_hat)[0]:.2f}")

# autoboost
bo = optimizer.BoostingOptimizer(initial_model=lightgbm.LGBMRegressor(), scorer=mse_scorer, n_folds=nfolds,
                                 verbose=verbose)
clf = bo.fit(x_train, y_train)
y_test_hat = clf.best_estimator_.predict(x_test)
print(f"lightgbm autoboost: {mean_squared_error(y_test, y_test_hat):.2f}")
print(f"lightgbm autoboost: {pearsonr(np.ravel(y_test.values), y_test_hat)[0]:.2f}")

f, ax = plt.subplots(1)
ax.scatter(y_test_hat, y_test)
ax.set(xlabel="predicted", ylabel="observed")
sns.despine()
plt.show()
