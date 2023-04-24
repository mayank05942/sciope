from sciope.features.feature_extraction import generate_tsfresh_features
from tsfresh.feature_extraction.settings import EfficientFCParameters
import numpy as np
import pytest
import pandas as pd


def test_generate_tsfresh_features():
    x = np.random.randn(2, 2, 100)

    # Convert x to a list of lists of pandas.Series with DatetimeIndex
    # x = [[pd.Series(x[i, j, :], index=pd.date_range(start='2000-01-01', periods=x.shape[2], freq='D'))
    #       for j in range(x.shape[1])] for i in range(x.shape[0])]

    features = EfficientFCParameters()
    test = generate_tsfresh_features(x, features)
    assert test.shape == (2, 1518)


