"""Shared config and helpers (imported by train.py and app.py so saved models load correctly)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DIABETES_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
DIABETES_TARGET = "Outcome"
# In the PIMA dataset a 0 in these columns means "not measured", not a real value
DIABETES_ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

HEART_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
HEART_TARGET = "target"


def clean_diabetes(df):
    """Turn impossible zeros into NaN so the imputer can fill them."""
    df = df.copy()
    df[DIABETES_ZERO_MISSING] = df[DIABETES_ZERO_MISSING].replace(0, np.nan)
    return df


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip outliers to [Q1 - k*IQR, Q3 + k*IQR] on continuous columns only.
    Columns with <= 10 distinct values (binary / categorical codes) are left untouched."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        self.mask_ = np.array([len(np.unique(c[~np.isnan(c)])) > 10 for c in X.T])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, self.mask_] = np.clip(X[:, self.mask_], self.lower_[self.mask_], self.upper_[self.mask_])
        return X