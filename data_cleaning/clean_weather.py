import numpy as np
import pandas as pd
from utils import (
    build_station2_location,
    get_calculated_sun_times,
    to_lst_string,
    fix_inconsistencies,
    parse_codes
)

def clean_weather(input_path, output_path):
    weather = pd.read_csv(input_path)
    weather["Date"] = pd.to_datetime(weather["Date"])

    weather["year"] = weather["Date"].dt.year
    weather["month"] = weather["Date"].dt.month
    weather["week"] = weather["Date"].dt.isocalendar().week.astype(int)
    weather["dayofyear"] = weather["Date"].dt.dayofyear

    num_cols = [
        "Tavg","Depart","WetBulb","Heat","Cool",
        "SnowFall","PrecipTotal","StnPressure",
        "SeaLevel","AvgSpeed"
    ]

    weather[num_cols] = (
        weather[num_cols]
        .astype(str)
        .replace({"M": np.nan, "T": 0.001})
        .apply(pd.to_numeric, errors="coerce")
    )

    weather["PrecipTotal"] = weather["PrecipTotal"].fillna(0)
    weather["Tavg"] = (weather["Tmax"] + weather["Tmin"]) / 2
    weather.drop(columns=["Depart","SnowFall"], inplace=True)

    weather["Sunrise"] = weather["Sunrise"].replace("-", np.nan)
    weather["Sunset"] = weather["Sunset"].replace("-", np.nan)

    station2 = build_station2_location()

    for idx, row in weather.iterrows():
        if pd.isna(row["Sunrise"]) or pd.isna(row["Sunset"]):
            sr, ss = get_calculated_sun_times(row["Date"].date(), station2)
            if pd.isna(row["Sunrise"]):
                weather.at[idx,"Sunrise"] = to_lst_string(sr)
            if pd.isna(row["Sunset"]):
                weather.at[idx,"Sunset"] = to_lst_string(ss)

    weather["Sunrise"] = weather["Sunrise"].apply(fix_inconsistencies).dt.time
    weather["Sunset"] = weather["Sunset"].apply(fix_inconsistencies).dt.time

    weather["day_length_min"] = (
        (
            pd.to_datetime(weather["Date"].astype(str) + " " + weather["Sunset"].astype(str))
            - pd.to_datetime(weather["Date"].astype(str) + " " + weather["Sunrise"].astype(str))
        ).dt.total_seconds() / 60
    )

    weather["CodeSum"] = weather["CodeSum"].fillna("")
    parsed = weather["CodeSum"].apply(parse_codes)
    weather = pd.concat([weather, pd.DataFrame(parsed.tolist())], axis=1)

    weather.drop(columns=["CodeSum","Depth","Water1"], inplace=True)
    weather.dropna(inplace=True)

    weather.to_csv(output_path, index=False)
