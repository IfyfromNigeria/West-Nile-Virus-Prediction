import pytz

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
REPORTS_DIR = "reports"

TIMEZONE = pytz.timezone("US/Central")

STATION_COORDS = {
    1: (41.9786, -87.9048),
    2: (41.786, -87.752)
}

EARTH_RADIUS_KM = 6371.0
RANDOM_STATE = 16
PCA_COMPONENTS = 25