# “””
West Nile Virus Outbreak Prediction — Pipeline Entry Point

Run this script from the project root to execute the full pipeline:

```
python main.py
```

The pipeline proceeds in the following order:
1. Clean raw datasets (train, weather, spray)
2. Merge train with the nearest weather station records
3. Append spray spatial-temporal features
4. Encode and engineer all model features
5. Time-aware train / test split
6. Scale + PCA dimensionality reduction
7. Train XGBoost and print evaluation metrics
“””

import pandas as pd

from data_cleaning.clean_train   import clean_train
from data_cleaning.clean_weather import clean_weather
from data_cleaning.clean_spray   import clean_spray

from feature_engineering.merge_datasets import merge_train_weather
from feature_engineering.spray_features import add_spray_features
from feature_engineering.encoding       import encode_features

from modeling.split         import split_data
from modeling.pca           import apply_pca
from modeling.train_xgboost import train_xgb

from src.config import (
RAW_DATA_DIR,
PROCESSED_DATA_DIR,
PCA_COMPONENTS,
setup_directories,
)

def main() -> None:

```
# ── 0. Ensure runtime directories exist ───────────────────────────────────
# data/ already exists in the cloned repo (raw CSVs committed there).
# This creates data/processed/ and reports/ if they are not yet present.
setup_directories()

# ── 1. Data cleaning ──────────────────────────────────────────────────────
print("\n[1/7] Cleaning datasets ...")
clean_train(
    f"{RAW_DATA_DIR}/train.csv",
    f"{PROCESSED_DATA_DIR}/train_cleaned.csv",
)
clean_weather(
    f"{RAW_DATA_DIR}/weather.csv",
    f"{PROCESSED_DATA_DIR}/weather_cleaned.csv",
)
clean_spray(
    f"{RAW_DATA_DIR}/spray.csv",
    f"{PROCESSED_DATA_DIR}/spray_cleaned.csv",
)

# ── 2. Merge train + weather ──────────────────────────────────────────────
print("\n[2/7] Merging train with weather ...")
train_weather = merge_train_weather(
    f"{PROCESSED_DATA_DIR}/train_cleaned.csv",
    f"{PROCESSED_DATA_DIR}/weather_cleaned.csv",
    f"{PROCESSED_DATA_DIR}/train_weather.csv",
)

# ── 3. Spray features ─────────────────────────────────────────────────────
print("\n[3/7] Adding spray features ...")
spray = pd.read_csv(
    f"{PROCESSED_DATA_DIR}/spray_cleaned.csv",
    parse_dates=["Date"],
)
train_weather = add_spray_features(train_weather, spray)
train_weather.to_csv(
    f"{PROCESSED_DATA_DIR}/train_weather_spray.csv", index=False
)

# ── 4. Feature encoding ───────────────────────────────────────────────────
print("\n[4/7] Encoding features ...")
data = encode_features(train_weather)

# ── 5. Train / test split ─────────────────────────────────────────────────
print("\n[5/7] Splitting data ...")
X_train, X_test, y_train, y_test = split_data(data)

# ── 6. Scaling + PCA ──────────────────────────────────────────────────────
print(f"\n[6/7] Applying PCA (n_components={PCA_COMPONENTS}) ...")
X_train_pca, X_test_pca, _ = apply_pca(X_train, X_test, PCA_COMPONENTS)

# ── 7. Model training & evaluation ────────────────────────────────────────
print("\n[7/7] Training XGBoost ...")
metrics = train_xgb(X_train_pca, y_train, X_test_pca, y_test)

print("\n✓ Pipeline complete.")
return metrics
```

if **name** == “**main**”:
main()
