import pandas as pd

def clean_train(input_path, output_path):
    train = pd.read_csv(input_path)
    train["Date"] = pd.to_datetime(train["Date"])

    train["year"] = train["Date"].dt.year
    train["month"] = train["Date"].dt.month
    train["week"] = train["Date"].dt.isocalendar().week.astype(int)
    train["dayofyear"] = train["Date"].dt.dayofyear

    train.drop(
        columns=[
            "Address","Street","AddressNumberAndStreet",
            "Block","AddressAccuracy"
        ],
        inplace=True
    )

    train_agg = (
        train.groupby(["Date","Trap","Species"], as_index=False)
        .agg({
            "NumMosquitos": "sum",
            "WnvPresent": "max",
            "Latitude": "first",
            "Longitude": "first",
            "year": "first",
            "month": "first",
            "week": "first",
            "dayofyear": "first"
        })
    )

    ordered_cols = [
        c for c in train_agg.columns
        if c not in ["NumMosquitos","WnvPresent"]
    ] + ["NumMosquitos","WnvPresent"]

    train_agg = train_agg[ordered_cols]
    train_agg.to_csv(output_path, index=False)
