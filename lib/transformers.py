import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def ftransformer_cut(X, **kwargs):
    assert isinstance(X, np.ndarray)
    if 'labels' not in kwargs:
        kwargs['labels'] = False
    assert kwargs['labels'] == False
    for jj in range(X.shape[1]):
        X[:, jj] = pd.cut(x=X[:, jj], **kwargs)
    return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
