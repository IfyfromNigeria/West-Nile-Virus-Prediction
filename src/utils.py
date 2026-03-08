import datetime
import re
import numpy as np
import pandas as pd
import pytz
from astral import LocationInfo
from astral.sun import sun
from geopy.distance import geodesic

from src.config import STATION_COORDS, TIMEZONE, EARTH_RADIUS_KM

# Weather phenomena codes used in CodeSum parsing 

PHENOMENA = [
“FC”, “TS”, “GR”, “RA”, “DZ”, “SN”, “SG”, “GS”, “PL”, “IC”,
“FG”, “BR”, “UP”, “HZ”, “FU”, “VA”, “DU”, “DS”, “PO”, “SA”,
“SS”, “PY”, “SQ”, “DR”, “SH”, “FZ”, “MI”, “PR”, “BC”, “BL”, “VC”,
]

# Station helpers

def nearest_station(lat: float, lon: float) -> int:
# Return the station number (1 or 2) closest to (lat, lon).
d1 = geodesic((lat, lon), STATION_COORDS[1]).km
d2 = geodesic((lat, lon), STATION_COORDS[2]).km
return 1 if d1 < d2 else 2

def build_station2_location() -> LocationInfo:
# Build an astral LocationInfo object for Station 2.
lat, lon = STATION_COORDS[2]
return LocationInfo(“Station_2”, “Chicago”, “US/Central”, lat, lon)

# Sunrise / sunset helpers

def get_calculated_sun_times(date, location: LocationInfo):
"""
Return (sunrise, sunset) as timezone-aware datetimes for *date*
at *location*, expressed in US/Central time.
"""
s = sun(location.observer, date=date, tzinfo=TIMEZONE)
return s[“sunrise”], s[“sunset”]

def to_lst_string(dt) -> str:
"""
Convert an aware datetime to a Local Standard Time HHMM string,
undoing any DST offset that astral may have applied.
"""
if dt.dst() != datetime.timedelta(0):
dt = dt - datetime.timedelta(hours=1)
return dt.strftime(”%H%M”)

def fix_inconsistencies(val: str):
"""
Parse a HHMM string into a pandas Timestamp, correcting the
edge-case where minutes equal ‘60’ (e.g. ‘0560’ → ‘0600’).
Returns pd.NaT on failure.
"""
try:
if str(val).endswith(“60”):
hour = int(str(val)[:2]) + 1
hour = 0 if hour >= 24 else hour
val  = f”{hour:02d}00”
return pd.to_datetime(val, format=”%H%M”)
except (ValueError, TypeError):
return pd.NaT

# CodeSum parsing

def parse_codes(codestr: str) -> dict:
"""
Parse a CodeSum string into binary indicator columns for each
weather phenomenon plus heavy / light intensity flags.
"""
s = str(codestr).upper()

output = {f"is_{p}": 0 for p in PHENOMENA}
output["intensity_heavy"] = int("+" in s)
output["intensity_light"] = int("-" in s)

cleaned = s.replace("+", " ").replace("-", " ")
for p in PHENOMENA:
    if re.search(rf"\b{p}\b", cleaned) or p in cleaned:
        output[f"is_{p}"] = 1

return output

# Unit conversion helpers

def km_to_radians(km: float) -> float:
# Convert kilometres to radians on the Earth’s surface.
return km / EARTH_RADIUS_KM

def fahrenheit_to_celsius(series: pd.Series) -> pd.Series:
return ((series - 32) * 5 / 9).round(1)

def mph_to_kmh(series: pd.Series) -> pd.Series:
return (series * 1.60934).round(1)

def inches_to_mm(series: pd.Series) -> pd.Series:
return (series * 25.4).round(2)

def inhg_to_hpa(series: pd.Series) -> pd.Series:
return (series * 33.8639).round(1)
