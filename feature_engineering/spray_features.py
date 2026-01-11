import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from utils import km_to_radians

def add_spray_features(train_weather, spray):
    spray_rad = np.radians(spray[["Latitude","Longitude"]].values)

    def has_spray(lat, lon, date, lag, radius):
        mask = (spray["Date"] <= date) & (spray["Date"] >= date - pd.Timedelta(days=lag))
        if not mask.any():
            return 0
        coords = spray_rad[mask.values]
        tree = BallTree(coords, metric="haversine")
        inds = tree.query_radius(
            np.radians([[lat, lon]]),
            r=km_to_radians(radius)
        )
        return int(len(inds[0]) > 0)

    for lag in [3,7,14]:
        train_weather[f"spray_within_{lag}d_1km"] = train_weather.apply(
            lambda r: has_spray(
                r["Latitude"], r["Longitude"], r["Date"], lag, 1
            ),
            axis=1
        )

    return train_weather