import pandas as pd

def clean_spray(input_path: str, output_path: str) -> pd.DataFrame:
    “””
    Load and clean the raw spray dataset.
    
    ```
    Steps
    -----
    - Parse dates.
    - Drop the Time column (not used in modelling).
    - Drop rows with missing values.
    - Remove duplicate records.
    
    Note: Time is dropped *before* dropna so that missing Time values
    do not cause valid spray location rows to be discarded.
    
    Parameters
    ----------
    input_path  : path to raw spray.csv
    output_path : path where the cleaned CSV will be saved
    
    Returns
    -------
    Cleaned spray DataFrame.
    """
    spray = pd.read_csv(input_path)
    spray["Date"] = pd.to_datetime(spray["Date"])
    
    # Drop Time first — it often contains nulls and is not needed
    spray.drop(columns=["Time"], inplace=True)
    
    spray.dropna(inplace=True)
    spray.drop_duplicates(inplace=True)
    
    spray.to_csv(output_path, index=False)
    return spray
