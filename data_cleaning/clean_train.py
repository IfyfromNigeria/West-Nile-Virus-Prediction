import pandas as pd

def clean_train(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load and clean the raw train dataset.
    Steps
    -----
    - Parse dates and extract temporal components.
    - Drop address-related columns that add no predictive value.
    - Aggregate duplicate trap / date / species rows by summing mosquito
      counts and taking the max WnvPresent label.
    
    Parameters
    ----------
    input_path  : path to raw train.csv
    output_path : path where the cleaned CSV will be saved
    
    Returns
    -------
    Cleaned and aggregated DataFrame.
    """
    train = pd.read_csv(input_path)
    train["Date"] = pd.to_datetime(train["Date"])
    
    # Temporal features
    train["year"]      = train["Date"].dt.year
    train["month"]     = train["Date"].dt.month
    train["week"]      = train["Date"].dt.isocalendar().week.astype(int)
    train["dayofyear"] = train["Date"].dt.dayofyear
    
    # Drop high-cardinality address columns
    train.drop(
        columns=["Address", "Street", "AddressNumberAndStreet",
                 "Block", "AddressAccuracy"],
        inplace=True,
    )
    
    # Aggregate duplicate trap/date/species records
    train_agg = (
        train.groupby(["Date", "Trap", "Species"], as_index=False)
        .agg(
            NumMosquitos=("NumMosquitos", "sum"),
            WnvPresent=("WnvPresent", "max"),
            Latitude=("Latitude", "first"),
            Longitude=("Longitude", "first"),
            year=("year", "first"),
            month=("month", "first"),
            week=("week", "first"),
            dayofyear=("dayofyear", "first"),
        )
    )
    
    # Reorder so labels sit at the end
    ordered = [
        c for c in train_agg.columns
        if c not in ("NumMosquitos", "WnvPresent")
    ] + ["NumMosquitos", "WnvPresent"]
    train_agg = train_agg[ordered]
    
    train_agg.to_csv(output_path, index=False)
    return train_agg
