import datetime
import numpy as np
import pandas as pd
import re
from astral import LocationInfo
from astral.sun import sun
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
from config import STATION_COORDS, TIMEZONE, EARTH_RADIUS_KM

PHENOMENA = [
    "FC","TS","GR","RA","DZ","SN","SG","GS","PL","IC","FG","BR","UP",
    "HZ","FU","VA","DU","DS","PO","SA","SS","PY","SQ","DR","SH",
    "FZ","MI","PR","BC","BL","VC"
]

def nearest_station(lat, lon):
    d1 = geodesic((lat, lon), STATION_COORDS[1]).km
    d2 = geodesic((lat, lon), STATION_COORDS[2]).km
    return 1 if d1 < d2 else 2

def build_station2_location():
    return LocationInfo(
        "Station_2",
        "Chicago",
        "US/Central",
        STATION_COORDS[2][0],
        STATION_COORDS[2][1]
    )

def get_calculated_sun_times(date, location):
    s = sun(location.observer, date=date, tzinfo=TIMEZONE)
    return s["sunrise"], s["sunset"]

def to_lst_string(dt):
    if dt.dst() != datetime.timedelta(0):
        dt = dt - datetime.timedelta(hours=1)
    return dt.strftime("%H%M")

def fix_inconsistencies(val):
    if val.endswith("60"):
        hour = int(val[:2]) + 1
        hour = 0 if hour >= 24 else hour
        val = f"{hour:02d}00"
    try:
        return pd.to_datetime(val, format="%H%M")
    except:
        return pd.NaT

def parse_codes(codestr):
    s = str(codestr).upper()
    output = {f"is_{p}": 0 for p in PHENOMENA}
    output["intensity_heavy"] = int("+" in s)
    output["intensity_light"] = int("-" in s)
    cleaned = s.replace("+", " ").replace("-", " ")
    for p in PHENOMENA:
        if re.search(rf"\b{p}\b", cleaned) or p in cleaned:
            output[f"is_{p}"] = 1
    return output

def km_to_radians(km):
    return km / EARTH_RADIUS_KM