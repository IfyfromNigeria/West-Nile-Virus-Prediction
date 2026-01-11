import pandas as pd

def clean_spray(input_path, output_path):
    spray = pd.read_csv(input_path)
    spray["Date"] = pd.to_datetime(spray["Date"])
    spray.dropna(inplace=True)
    spray.drop(columns=["Time"], inplace=True)
    spray.drop_duplicates(inplace=True)
    spray.to_csv(output_path, index=False)