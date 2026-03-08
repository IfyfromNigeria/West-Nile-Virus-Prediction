import numpy as np
import pandas as pd

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering, encoding, and column pruning to the fully
    merged dataset (train + weather + spray features).
    
    ```
    Steps
    -----
    1. Sort by date to ensure rolling windows are computed correctly.
    2. Compute 14-day rolling average for Tavg and PrecipTotal (per station).
    3. Compute 7-day daylight change.
    4. Compute relative humidity from DewPoint and Tavg (Magnus formula).
    5. Compute 30-day cumulative cooling degree days (per station).
    6. Cyclically encode month (sin / cos).
    7. One-hot encode Species.
    8. Drop columns that are redundant, leak the target, or have low
       biological impact.
    
    Parameters
    ----------
    df : merged DataFrame containing all raw and spray features
    
    Returns
    -------
    Model-ready DataFrame (copy — original is not mutated).
    """
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data.sort_values("Date", inplace=True)
    
    # ── Rename duplicated _x / _y columns from the merge
    rename_map = {
        "year_x": "year", "month_x": "month",
        "week_x": "week", "dayofyear_x": "dayofyear",
    }
    data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns},
                inplace=True)
    dup_cols = ["year_y", "month_y", "week_y", "dayofyear_y"]
    data.drop(columns=[c for c in dup_cols if c in data.columns], inplace=True)
    
    # ── Temporal rolling features (grouped by station for correctness)
    data["Tavg_14d_avg"] = (
        data.groupby("Station")["Tavg"]
        .transform(lambda x: x.rolling(14, min_periods=1).mean().shift(1))
    )
    
    data["PrecipTotal_14d_avg"] = (
        data.groupby("Station")["PrecipTotal"]
        .transform(lambda x: x.rolling(14, min_periods=1).mean().shift(1))
    )
    
    data["daylight_change_7d"] = (
        data["day_length_min"] - data["day_length_min"].shift(7)
    )
    
    data["cooling_30d_cum"] = (
        data.groupby("Station")["Cool"]
        .transform(lambda x: x.rolling(30, min_periods=1).sum().shift(1))
    )
    
    # Back-fill the leading NaNs introduced by shifting
    backfill_cols = [
        "Tavg_14d_avg", "PrecipTotal_14d_avg",
        "daylight_change_7d", "cooling_30d_cum",
    ]
    data[backfill_cols] = data[backfill_cols].bfill()
    
    # ── Relative humidity via Magnus formula
    # Note: Tavg and DewPoint must still be in Celsius at this point
    a, b = 17.625, 243.04
    data["RH_percent"] = 100 * (
        np.exp(a * data["DewPoint"] / (b + data["DewPoint"]))
        / np.exp(a * data["Tavg"]    / (b + data["Tavg"]))
    )
    
    # ── Cyclical month encoding
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    
    # ── One-hot encode Species
    data = pd.get_dummies(data, columns=["Species"], drop_first=True)
    species_cols = [c for c in data.columns if c.startswith("Species_")]
    data[species_cols] = data[species_cols].astype(int)
    
    # ── Drop columns 
    # Raw temperatures  — replaced by rolling avg and RH
    # month / week      — encoded cyclically above
    # Sunrise / Sunset  — day_length_min captures the signal
    # PrecipTotal       — replaced by 14-day rolling avg
    # Trap              — Latitude / Longitude carry the spatial info
    # ResultSpeed       — AvgSpeed is preferred
    # DewPoint          — replaced by RH_percent
    # SeaLevel / StnPressure / ResultDir — low biological impact
    # Heat / Cool       — rolling cumulative version retained; raw dropped
    # NumMosquitos      — target leakage
    # Date / closest_station / Station — not model features
    # Tavg              — replaced by rolling avg and RH
    drop_cols = [
        "Tmin", "Tmax", "month", "week",
        "Sunrise", "Sunset",
        "PrecipTotal",
        "Trap",
        "ResultSpeed",
        "DewPoint",
        "SeaLevel", "StnPressure", "ResultDir",
        "Heat", "Cool",
        "NumMosquitos",
        "Date", "closest_station", "Station",
        "Tavg",
    ]
    data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)
    
    # ── Clamp negative days_since_last_spray sentinel values
    data.loc[data["days_since_last_spray"] < 0, "days_since_last_spray"] = 999
    
    return data
