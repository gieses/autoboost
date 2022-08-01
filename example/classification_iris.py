"""
Demonstration on how to use classification data with autoboost.

One has to take special care to correctly parameterize the scorer and encoding the target variable.
"""
import warnings

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer

from autoboost import optimizer

warnings.simplefilter(action='ignore', category=FutureWarning)


def prepare_iris():
    df = sns.load_dataset("iris")

    X = df.filter(regex="^(?!species)")
    y = df.filter(regex="species")

    label_encoder = preprocessing.LabelEncoder()
    y = np.ravel(label_encoder.fit_transform(y.values))
    return X, y


verbose = 0
nfolds = 3

# get data
print("getting data ready ...")
X, y = prepare_iris()

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=1)
clf_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

# %% default
xgb_default = xgboost.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
xgb_default.fit(x_train, y_train)
yhat_test_xgb_default = xgb_default.predict(x_test)
print(f"xgboost regular: {accuracy_score(yhat_test_xgb_default, y_test):.2f}")

# %% xgboost
xgboost_initial = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
bo = optimizer.BoostingOptimizer(initial_model=xgboost_initial, scorer=clf_scorer, n_folds=nfolds, verbose=verbose)
clf = bo.fit(x_train, y_train)
y_test_hat = clf.best_estimator_.predict(x_test)
print(f"xgboost optimized: {accuracy_score(y_test_hat, y_test):.2f}")

# %% lgbm
bo = optimizer.BoostingOptimizer(initial_model=lightgbm.LGBMClassifier(), scorer=clf_scorer, verbose=verbose,
                                 n_folds=nfolds)
clf = bo.fit(x_train, y_train)
y_test_hat = clf.best_estimator_.predict(x_test)
print(f"lightgbm optimized: {accuracy_score(y_test_hat, y_test):.2f}")

# %% GBC
bo = optimizer.BoostingOptimizer(initial_model=GradientBoostingClassifier(), scorer=clf_scorer, verbose=verbose,
                                 n_folds=nfolds)
clf = bo.fit(x_train, y_train)
y_test_hat = clf.best_estimator_.predict(x_test)
print(f"GradientBoostingClassifier optimized: {accuracy_score(y_test_hat, y_test):.2f}")

# a semi-random visual
cm = confusion_matrix(y_test, y_test_hat)
f, ax = plt.subplots(figsize=(8, 5))
cmp = ConfusionMatrixDisplay(cm, display_labels=["class_1", "class_2", "class_3"])
cmp.plot()
plt.show()
