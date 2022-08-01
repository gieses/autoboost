import warnings

import lightgbm
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from autoboost import const
from autoboost.optimizer import BoostingOptimizer

warnings.simplefilter(action='ignore', category=FutureWarning)


def test_optimizer():
    bo = BoostingOptimizer()
    bo.initial_param_dic = [{'n_estimators': [50, 100, 150],
                             'n_trees': [50, 100, 150]}]
    tasks = bo.get_optimization_tasks()
    assert tasks == [{'n_estimators': [50, 100, 150]}]

    bo = BoostingOptimizer(initial_params_dict="small")
    assert bo.initial_param_dic == const.UNIVERSAL_PARAMS_SMALL

    bo = BoostingOptimizer(initial_params_dict="default")
    assert bo.initial_param_dic == const.UNIVERSAL_PARAMS_DEFAULT

    with pytest.raises(ValueError):
        _ = BoostingOptimizer(initial_params_dict="supermegalarge")


def test_fit_xgboost():
    x = pd.DataFrame()
    x["var1"] = [2, 4, 6, 8, 10, 12, 14]
    x["var2"] = np.sqrt([2, 4, 6, 8, 10, 12, 14]) + np.random.normal(size=len(x["var1"]))
    y = x["var1"] ** + x["var2"] + np.random.normal(size=len(x["var1"]))

    bo = BoostingOptimizer(verbose=True, n_folds=3)
    bo.initial_param_dic = [{'n_estimators': [1, 3, 5]}]
    res = bo.fit(x, y)

    assert len(pd.concat(bo.cv_results)) > 0
    assert res.best_params_


def test_fit_xgboost_custom():
    x = pd.DataFrame()
    x["var1"] = [2, 4, 6, 8, 10, 12, 14]
    x["var2"] = np.sqrt([2, 4, 6, 8, 10, 12, 14]) + np.random.normal(size=len(x["var1"]))
    y = x["var1"] ** + x["var2"] + np.random.normal(size=len(x["var1"]))

    bo = BoostingOptimizer(verbose=True, initial_params_dict=[{'n_estimators': [1, 3, 5]}], n_folds=3)
    res = bo.fit(x, y)

    assert len(pd.concat(bo.cv_results)) > 0
    assert res.best_params_


def test_fit_lightgbm():
    x = pd.DataFrame()
    x["var1"] = [2, 4, 6, 8, 10, 12, 14]
    x["var2"] = np.sqrt([2, 4, 6, 8, 10, 12, 14]) + np.random.normal(size=len(x["var1"]))
    y = x["var1"] ** + x["var2"] + np.random.normal(size=len(x["var1"]))

    bo = BoostingOptimizer(lightgbm.LGBMRegressor(), verbose=True, n_folds=3)
    bo.initial_param_dic = [{'n_estimators': [1, 3, 5]}]
    res = bo.fit(x, y)

    assert len(pd.concat(bo.cv_results)) > 0
    assert res.best_params_


def test_fit_sklearn():
    x = pd.DataFrame()
    x["var1"] = [2, 4, 6, 8, 10, 12, 14]
    x["var2"] = np.sqrt([2, 4, 6, 8, 10, 12, 14]) + np.random.normal(size=len(x["var1"]))
    y = x["var1"] ** + x["var2"] + np.random.normal(size=len(x["var1"]))

    bo = BoostingOptimizer(GradientBoostingRegressor(), verbose=True, n_folds=3)
    bo.initial_param_dic = [{'n_estimators': [1, 3, 5]}]
    res = bo.fit(x, y)

    assert len(pd.concat(bo.cv_results)) > 0
    assert res.best_params_


def test_optimizer_valid_model():
    with pytest.raises(AssertionError) as _:
        _ = BoostingOptimizer(initial_model=RandomForestClassifier)


def test_optimizer_valid_loss():
    with pytest.raises(AssertionError) as _:
        _ = BoostingOptimizer(min_loss=0.0)


def test_optimizer_valid_nfolds():
    with pytest.raises(AssertionError) as _:
        _ = BoostingOptimizer(n_folds=0)


def test_define_new_param_borders():
    param = "n_estimators"
    valid_range = {"n_estimators": {'max': 500, 'min': 1, 'type': 'int'}}
    new_ar = BoostingOptimizer.define_new_param_borders(param,
                                                        [30, 50, 70, 100, 150, 200, 300], 2, valid_range)
    expected_ar = np.array([60, 70, 85])
    assert np.allclose(new_ar, expected_ar)

    params = [30, 50, 70]
    new_ar = BoostingOptimizer.define_new_param_borders(param, params, 0, valid_range)
    expected_ar = np.array([10, 30, 40])
    assert np.allclose(new_ar, expected_ar)

    params = [30, 50, 70]
    new_ar = BoostingOptimizer.define_new_param_borders(param, params, 2, valid_range)
    expected_ar = np.array([60, 70, 95])
    assert np.allclose(new_ar, expected_ar)

    params = [1, 5, 7]
    new_ar = BoostingOptimizer.define_new_param_borders(param, params, 0, valid_range)
    expected_ar = np.array([1, 3])
    assert np.allclose(new_ar, expected_ar)

    params = [480, 490, 500]
    new_ar = BoostingOptimizer.define_new_param_borders(param, params, 2, valid_range)
    expected_ar = np.array([495, 500])
    assert np.allclose(new_ar, expected_ar)
