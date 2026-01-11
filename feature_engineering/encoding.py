import numpy as np
import pandas as pd

def encode_features(df):
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df.drop(
        columns=[
            "Tmin","Tmax","month","week","Sunrise","Sunset",
            "PrecipTotal","Trap","ResultSpeed","DewPoint",
            "SeaLevel","StnPressure","ResultDir","Heat",
            "Cool","NumMosquitos","closest_station","Station","Date"
        ],
        inplace=True
    )

    df = pd.get_dummies(df, columns=["Species"], drop_first=True)
    df[df.columns] = df[df.columns].apply(lambda x: x.astype(int) if x.dtype == bool else x)

    df.loc[df["days_since_last_spray"] < 0, "days_since_last_spray"] = 999
    return df