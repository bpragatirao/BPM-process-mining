import pm4py
import pandas as pd

def convert_xes_to_csv(xes_path, output_csv):
    print("Reading XES file...")
    log = pm4py.read_xes(xes_path)

    print("Converting to DataFrame...")
    df = pm4py.convert_to_dataframe(log)

    # Keep only required columns
    df = df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp"
    })

    df = df[["case_id", "activity", "timestamp"]]
    df = df.sort_values(["case_id", "timestamp"])

    print("Saving CSV...")
    df.to_csv(output_csv, index=False)

    print("Conversion complete.")


if __name__ == "__main__":
    convert_xes_to_csv(
        "data/raw/BPI Challenges 2017.xes",
        "data/raw/BPI_Challenges_2017.csv"
    )
