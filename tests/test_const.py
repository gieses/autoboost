from autoboost import const


def test_params():
    assert len(const.UNIVERSAL_PARAMS_SMALL) != 0
    assert len(const.UNIVERSAL_PARAMS_DEFAULT) != 0


def test_mse():
    mse = const.get_mse()
    assert mse._sign == -1


def test_rmse():
    rmse = const.get_rmse()
    assert rmse._sign == -1
