import pandas as pd
from utils import nearest_station

def merge_train_weather(train_path, weather_path, output_path):
    train = pd.read_csv(train_path, parse_dates=["Date"])
    weather = pd.read_csv(weather_path, parse_dates=["Date"])

    train["closest_station"] = train.apply(
        lambda r: nearest_station(r["Latitude"], r["Longitude"]),
        axis=1
    )

    train_weather = train.merge(
        weather,
        left_on=["Date","closest_station"],
        right_on=["Date","Station"],
        how="left"
    )

    train_weather.dropna(inplace=True)
    train_weather.to_csv(output_path, index=False)

    return train_weather