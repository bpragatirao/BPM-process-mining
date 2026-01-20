import pandas as pd
import pm4py

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.importer.mxml import importer as mxml_importer


def load_event_log(file_path):

    if file_path.endswith(".mxml"):
        event_log = mxml_importer.apply(file_path)
    elif file_path.endswith(".xes"):
        event_log = xes_importer.apply(file_path)
    else:
        raise ValueError("Unsupported file format. Use .mxml or .xes")

    df = pm4py.convert_to_dataframe(event_log)
    df = df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp"
    })

    df = df[["case_id", "activity", "timestamp"]]

    df = df.sort_values(by=["case_id", "timestamp"])

    return df


if __name__ == "__main__":
    df = load_event_log("data/raw/CreditRequirements.mxml")
    print(df.head())
    print("Total cases:", df["case_id"].nunique())
