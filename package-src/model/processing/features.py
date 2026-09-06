import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols):
        self.date_cols = date_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.date_cols:
            if col in X_out.columns:
                dt_series = pd.to_datetime(X_out[col], errors="coerce")
                X_out[f"{col}_month"] = dt_series.dt.month
                X_out[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                X_out[f"{col}_quarter"] = dt_series.dt.quarter
                X_out.drop(columns=[col], inplace=True)
        return X_out