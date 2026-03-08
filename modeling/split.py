from src.config import RANDOM_STATE

def split_data(data):
    train = data[data["year"] < 2013]
    test = data[data["year"] == 2013]

    train = train.sample(frac=1, random_state=RANDOM_STATE)

    X_train = train.drop(columns=["WnvPresent","year"])
    y_train = train["WnvPresent"]

    X_test = test.drop(columns=["WnvPresent","year"])
    y_test = test["WnvPresent"]


    return X_train, X_test, y_train, y_test

import pandas as pd

from src.config import RANDOM_STATE, TEST_YEAR

def split_data(data: pd.DataFrame):
“””
Perform a time-aware train / test split to prevent temporal leakage.
All years strictly before TEST_YEAR are used for training; TEST_YEAR
itself forms the held-out test set.  The training set is shuffled so
that XGBoost does not see any ordering artefacts.

Parameters
----------
data : fully encoded DataFrame that still contains 'year' and
       'WnvPresent' columns

Returns
-------
X_train, X_test, y_train, y_test
"""
train_set = data[data["year"] <  TEST_YEAR].copy()
test_set  = data[data["year"] == TEST_YEAR].copy()

if train_set.empty:
    raise ValueError(
        f"Training set is empty. No rows found with year < {TEST_YEAR}."
    )
if test_set.empty:
    raise ValueError(
        f"Test set is empty. No rows found with year == {TEST_YEAR}. "
        "Check TEST_YEAR in src/config.py."
    )

train_set = train_set.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X_train = train_set.drop(columns=["WnvPresent", "year"])
y_train = train_set["WnvPresent"]

X_test  = test_set.drop(columns=["WnvPresent", "year"])
y_test  = test_set["WnvPresent"]

print(
    f"[split_data] Train: {len(X_train)} rows  |  "
    f"Test: {len(X_test)} rows  |  "
    f"Positive rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}"
)

return X_train, X_test, y_train, y_test
