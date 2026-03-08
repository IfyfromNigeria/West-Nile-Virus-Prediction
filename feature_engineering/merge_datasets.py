import pandas as pd
from src.utils import nearest_station

def merge_train_weather(
    train_path: str,
    weather_path: str,
    output_path: str,
    ) -> pd.DataFrame:
    """
    Merge the cleaned train dataset with the cleaned weather dataset.
    
    Each trap record is matched to its nearest weather station rather than
    averaging across both stations, preserving localised weather signals.
    
    Steps
    -----
    - Assign each trap record to its nearest station (1 or 2) by geodesic
      distance.
    - Left-join train with weather on (Date, nearest_station == Station).
    - Log how many rows were successfully matched before dropping nulls.
    - Drop unmatched rows.
    
    Parameters
    ----------
    train_path   : path to cleaned train CSV
    weather_path : path to cleaned weather CSV
    output_path  : path where the merged CSV will be saved
    
    Returns
    -------
    Merged DataFrame with weather features appended to each trap record.
    """
    train   = pd.read_csv(train_path,   parse_dates=["Date"])
    weather = pd.read_csv(weather_path, parse_dates=["Date"])
    
    # Assign nearest station to every trap row
    train["closest_station"] = train.apply(
        lambda r: nearest_station(r["Latitude"], r["Longitude"]),
        axis=1,
    )
    
    train_weather = train.merge(
        weather,
        left_on=["Date", "closest_station"],
        right_on=["Date", "Station"],
        how="left",
    )
    
    # Log match rate so data loss is visible
    total   = len(train_weather)
    matched = train_weather["Tavg"].notna().sum()
    unmatched = total - matched
    if unmatched > 0:
        print(
            f"[merge_train_weather] WARNING: {unmatched} / {total} rows "
            f"could not be matched to a weather record and will be dropped."
        )
    else:
        print(f"[merge_train_weather] All {total} rows matched successfully.")
    
    train_weather.dropna(inplace=True)
    train_weather.to_csv(output_path, index=False)
    return train_weather
