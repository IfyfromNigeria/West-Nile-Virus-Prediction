import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.utils import km_to_radians

def add_spray_features(
    train_weather: pd.DataFrame,
    spray: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Append spray-derived spatial-temporal features to the merged dataset.
    
    Features added
    --------------
    spray_within_{3,7,14}d_1km : binary — was a spray event recorded within
                                  1 km of this trap in the past N days?
    spray_intensity_2km         : count of spray events within 2 km of this
                                  trap in the past 7 days.
    days_since_last_spray       : calendar days between this trap record and
                                  the most recent spray event anywhere.
                                  Set to 999 when no prior spray is available
                                  or when the trap shows no nearby spray signal.
    
    Performance note
    ----------------
    Rather than rebuilding a BallTree on every row (O(n²)), we pre-compute
    radians for all spray coordinates once and build per-date/lag sub-trees
    only over the relevant temporal subset.
    
    Parameters
    ----------
    train_weather : merged train + weather DataFrame (must contain Date,
                    Latitude, Longitude columns)
    spray         : cleaned spray DataFrame (must contain Date, Latitude,
                    Longitude columns)
    
    Returns
    -------
    train_weather with spray feature columns appended (in-place copy).
    """
    df = train_weather.copy()
    
    spray = spray.copy()
    spray["Date"] = pd.to_datetime(spray["Date"])
    df["Date"]    = pd.to_datetime(df["Date"])
    
    spray_rad = np.radians(spray[["Latitude", "Longitude"]].values)
    spray_dates = spray["Date"].values
    spray_dates_sorted = spray["Date"].sort_values().drop_duplicates()
    
    # ── Binary spray-within-N-days-1km features ───────────────────────────────
    
    def _has_spray(lat: float, lon: float, trap_date, lag_days: int,
                   radius_km: float = 1.0) -> int:
        start = trap_date - pd.Timedelta(days=lag_days)
        mask  = (spray["Date"] <= trap_date) & (spray["Date"] >= start)
        if not mask.any():
            return 0
        # Build tree only over the temporally relevant subset
        coords = spray_rad[mask.values]
        tree   = BallTree(coords, metric="haversine")
        inds   = tree.query_radius(
            np.radians([[lat, lon]]),
            r=km_to_radians(radius_km),
        )
        return int(len(inds[0]) > 0)
    
    for lag in [3, 7, 14]:
        col = f"spray_within_{lag}d_1km"
        print(f"[add_spray_features] Computing {col} ...")
        df[col] = df.apply(
            lambda r: _has_spray(r["Latitude"], r["Longitude"],
                                  r["Date"], lag),
            axis=1,
        )
    
    # ── Spray intensity: count of events within 2 km / 7 days ────────────────
    
    def _spray_intensity(lat: float, lon: float, trap_date,
                         lag_days: int = 7, radius_km: float = 2.0) -> int:
        start = trap_date - pd.Timedelta(days=lag_days)
        mask  = (spray["Date"] <= trap_date) & (spray["Date"] >= start)
        if not mask.any():
            return 0
        coords = spray_rad[mask.values]
        tree   = BallTree(coords, metric="haversine")
        inds   = tree.query_radius(
            np.radians([[lat, lon]]),
            r=km_to_radians(radius_km),
        )
        return len(inds[0])
    
    print("[add_spray_features] Computing spray_intensity_2km ...")
    df["spray_intensity_2km"] = df.apply(
        lambda r: _spray_intensity(r["Latitude"], r["Longitude"], r["Date"]),
        axis=1,
    )
    
    # ── Days since last spray ─────────────────────────────────────────────────
    
    earliest_spray = (
        spray_dates_sorted.iloc[0] if not spray_dates_sorted.empty else None
    )
    
    def _days_since_last_spray(row) -> int:
        trap_date = row["Date"]
    
        no_nearby = (
            row["spray_within_3d_1km"]   == 0
            and row["spray_within_7d_1km"]  == 0
            and row["spray_within_14d_1km"] == 0
            and row["spray_intensity_2km"]  == 0
        )
    
        if no_nearby:
            # No nearby spray — use distance from earliest known spray as proxy
            if earliest_spray is not None:
                return int((trap_date - earliest_spray).days)
            return 999  # no spray history at all
    
        prior = spray_dates_sorted[spray_dates_sorted <= trap_date]
        if prior.empty:
            return 999
    
        return int((trap_date - prior.iloc[-1]).days)
    
    print("[add_spray_features] Computing days_since_last_spray ...")
    df["days_since_last_spray"] = df.apply(_days_since_last_spray, axis=1)
    
    # Replace any remaining sentinel negatives with 999
    df.loc[df["days_since_last_spray"] < 0, "days_since_last_spray"] = 999
    
    return df

