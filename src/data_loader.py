import pm4py
import pandas as pd

def load_event_log(file_path):
    event_log = pm4py.read_xes(file_path)

    df = pm4py.convert_to_dataframe(event_log)

    df = df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp"
    })

    df = df[["case_id", "activity", "timestamp"]]
    df = df.sort_values(by=["case_id", "timestamp"])

    return df
