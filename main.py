import pandas as pd
from src.data_cleaning.clean_train import clean_train
from src.data_cleaning.clean_weather import clean_weather
from src.data_cleaning.clean_spray import clean_spray
from src.feature_engineering.merge_datasets import merge_train_weather
from src.feature_engineering.spray_features import add_spray_features
from src.feature_engineering.encoding import encode_features
from src.modeling.split import split_data
from src.modeling.pca import apply_pca
from src.modeling.train_xgboost import train_xgb
from config import PCA_COMPONENTS

def main():
    clean_train("data/raw/train.csv", "data/processed/train_cleaned.csv")
    clean_weather("data/raw/weather.csv", "data/processed/weather_cleaned.csv")
    clean_spray("data/raw/spray.csv", "data/processed/spray_cleaned.csv")

    train_weather = merge_train_weather(
        "data/processed/train_cleaned.csv",
        "data/processed/weather_cleaned.csv",
        "data/processed/train_weather.csv"
    )

    spray = pd.read_csv("data/processed/spray_cleaned.csv", parse_dates=["Date"])
    train_weather = add_spray_features(train_weather, spray)

    train_weather.to_csv("data/processed/train_weather_spray.csv", index=False)

    data = encode_features(train_weather)

    X_train, X_test, y_train, y_test = split_data(data)
    X_train_pca, X_test_pca, _ = apply_pca(X_train, X_test, PCA_COMPONENTS)

    metrics = train_xgb(X_train_pca, y_train, X_test_pca, y_test)
    print(metrics)

if __name__ == "__main__":
    main()
