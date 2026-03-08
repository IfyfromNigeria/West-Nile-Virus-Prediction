import pytz
# ── Directory paths
RAW_DATA_DIR       = “data”
PROCESSED_DATA_DIR = “data/processed”
REPORTS_DIR        = “reports”

# Timezone

TIMEZONE = pytz.timezone(“US/Central”)

# Weather station coordinates (lat, lon) 

STATION_COORDS = {
1: (41.9786, -87.9048),
2: (41.786,  -87.752),
}

# Modelling constants 

EARTH_RADIUS_KM = 6371.0
RANDOM_STATE    = 16
PCA_COMPONENTS  = 25
TEST_YEAR       = 2013          # year used as held-out test set

import pytz
from pathlib import Path

# ── Directory paths ───────────────────────────────────────────────────────────

# Raw CSVs (train.csv, weather.csv, spray.csv) sit directly under data/

# as committed in the repo — no data/raw/ subdirectory exists.

RAW_DATA_DIR       = “data”
PROCESSED_DATA_DIR = “data/processed”   # created at runtime, not in repo
REPORTS_DIR        = “reports”           # created at runtime, not in repo

def setup_directories() -> None:
“””
Create directories that are needed at runtime but are not committed
to the repo (processed outputs and reports).

```
RAW_DATA_DIR (data/) already exists in the cloned repo and is
intentionally excluded here — it should never be recreated empty.

Safe to call multiple times — existing folders are left untouched.
"""
for path in [PROCESSED_DATA_DIR, REPORTS_DIR]:
    Path(path).mkdir(parents=True, exist_ok=True)
```

# Timezone
TIMEZONE = pytz.timezone(“US/Central”)

# Weather station coordinates (lat, lon)

STATION_COORDS = {
1: (41.9786, -87.9048),
2: (41.786,  -87.752),
}

# Modelling constants

EARTH_RADIUS_KM = 6371.0
RANDOM_STATE    = 16
PCA_COMPONENTS  = 25
TEST_YEAR       = 2013          # year used as held-out test set
