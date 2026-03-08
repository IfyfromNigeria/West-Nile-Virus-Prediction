import numpy as np
import pandas as pd

from src.utils import (
    build_station2_location,
    get_calculated_sun_times,
    to_lst_string,
    fix_inconsistencies,
    parse_codes,
    fahrenheit_to_celsius,
    mph_to_kmh,
    inches_to_mm,
    inhg_to_hpa,
    )

def clean_weather(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load and clean the raw weather dataset.
    Steps
    -----
    - Parse dates and extract temporal components.
    - Convert string-encoded missing values ('M', 'T') to numeric.
    - Impute missing Tavg from Tmax / Tmin.
    - Fill missing Station 2 sunrise / sunset using NOAA solar algorithm.
    - Compute day_length_min from sunrise and sunset.
    - Parse CodeSum into binary weather-phenomenon indicator columns.
    - Convert all measurements to SI / metric units.
    - Align Station 1 and Station 2 to share the same dates.
    - Drop rows with remaining missing values.
    
    Parameters
    ----------
    input_path  : path to raw weather.csv
    output_path : path where the cleaned CSV will be saved
    
    Returns
    -------
    Cleaned weather DataFrame.
    """
    weather = pd.read_csv(input_path)
    weather["Date"] = pd.to_datetime(weather["Date"])
    
    # Temporal features
    weather["year"]      = weather["Date"].dt.year
    weather["month"]     = weather["Date"].dt.month
    weather["week"]      = weather["Date"].dt.isocalendar().week.astype(int)
    weather["dayofyear"] = weather["Date"].dt.dayofyear
    
    # Numeric columns stored as strings
    num_cols = [
        "Tavg", "Depart", "WetBulb", "Heat", "Cool",
        "SnowFall", "PrecipTotal", "StnPressure", "SeaLevel", "AvgSpeed",
    ]
    weather[num_cols] = (
        weather[num_cols]
        .apply(lambda col: col.astype(str).str.strip())
        .replace({"M": np.nan, "T": 0.001})
        .apply(pd.to_numeric, errors="coerce")
    )
    
    # Trace precipitation treated as 0 after conversion
    weather["PrecipTotal"] = weather["PrecipTotal"].fillna(0)
    
    # Recompute Tavg from raw Tmax/Tmin (more reliable than the stored value)
    weather["Tavg"] = (weather["Tmax"] + weather["Tmin"]) / 2
    
    # Drop columns with too many missing values or low signal
    weather.drop(columns=["Depart", "SnowFall"], inplace=True)
    
    # ── Sunrise / Sunset ──────────────────────────────────────────────────────
    weather["Sunrise"] = weather["Sunrise"].replace("-", np.nan)
    weather["Sunset"]  = weather["Sunset"].replace("-", np.nan)
    
    # Station 2 has systemic missing values — fill via astronomical calculation
    station2 = build_station2_location()
    missing_mask = weather["Sunrise"].isna() | weather["Sunset"].isna()
    for idx in weather.index[missing_mask]:
        row = weather.loc[idx]
        sr, ss = get_calculated_sun_times(row["Date"].date(), station2)
        if pd.isna(row["Sunrise"]):
            weather.at[idx, "Sunrise"] = to_lst_string(sr)
        if pd.isna(row["Sunset"]):
            weather.at[idx, "Sunset"]  = to_lst_string(ss)
    
    # Parse HHMM strings → time objects
    weather["Sunrise"] = weather["Sunrise"].apply(fix_inconsistencies).dt.time
    weather["Sunset"]  = weather["Sunset"].apply(fix_inconsistencies).dt.time
    
    # Day-length in minutes
    date_str = weather["Date"].dt.strftime("%Y-%m-%d")
    weather["day_length_min"] = (
        (
            pd.to_datetime(date_str + " " + weather["Sunset"].astype(str))
            - pd.to_datetime(date_str + " " + weather["Sunrise"].astype(str))
        ).dt.total_seconds() / 60
    )
    
    # ── CodeSum → binary indicator columns
    weather["CodeSum"] = weather["CodeSum"].fillna("").astype(str)
    parsed = weather["CodeSum"].apply(parse_codes)
    weather = pd.concat(
        [weather, pd.DataFrame(parsed.tolist(), index=weather.index)],
        axis=1,
    )
    weather.drop(columns=["CodeSum", "Depth", "Water1"], inplace=True)
    
    # ── Drop remaining missing rows (< 0.5 % of dataset) 
    weather.dropna(inplace=True)
    
    # ── Unit conversions → metric 
    temp_cols = ["Tmax", "Tmin", "Tavg", "DewPoint", "WetBulb"]
    for col in temp_cols:
        weather[col] = fahrenheit_to_celsius(weather[col])
    
    weather["ResultSpeed"] = mph_to_kmh(weather["ResultSpeed"])
    weather["AvgSpeed"]    = mph_to_kmh(weather["AvgSpeed"])
    
    # Heat / Cool degree-days are Fahrenheit-based; rescale to Celsius-based
    for col in ["Heat", "Cool"]:
        weather[col] = (weather[col] * 5 / 9).round(1)
    
    weather["PrecipTotal"] = inches_to_mm(weather["PrecipTotal"])
    weather["StnPressure"] = inhg_to_hpa(weather["StnPressure"])
    
    # ── Align Station 1 & 2 to shared dates 
    dates1 = set(weather.loc[weather["Station"] == 1, "Date"])
    dates2 = set(weather.loc[weather["Station"] == 2, "Date"])
    common_dates = dates1.intersection(dates2)
    weather = weather.loc[weather["Date"].isin(common_dates)].copy()
    
    # Final safety pass
    weather.dropna(inplace=True)
    
    weather.to_csv(output_path, index=False)
    return weather
