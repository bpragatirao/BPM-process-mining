import pandas as pd
from sklearn.preprocessing import LabelEncoder

def generate_prefixes(df, max_prefix_length = 15, max_cases=3000):
    """
    Convert an event log DataFrame into
    prefix-label training samples.

    Input:
        df: DataFrame with columns
            [case_id, activity, timestamp]

    Output:
        X: list of activity prefixes
        y: list of next-activity labels
    """

    X = []  
    y = []  

    # Limit number of cases
    case_ids = df["case_id"].unique()[:max_cases]
    df = df[df["case_id"].isin(case_ids)]

    # Group events by case
    grouped = df.groupby("case_id")

    # Process each case independently
    for idx, (i, case_data) in enumerate(grouped):
        if idx >= max_cases:
            break
        # Ensure correct ordering
        case_data = case_data.sort_values("timestamp")
        # Extract activity sequence
        activities = case_data["activity"].tolist()

        MIN_PREFIX_LENGTH = 3

        # Generate prefixes
        for i in range(1, len(activities)):
            prefix = activities[:i]
            if len(prefix) < MIN_PREFIX_LENGTH:
                continue
            if len(prefix) > max_prefix_length:
                prefix = prefix[-max_prefix_length:]

            X.append(prefix)
            y.append(activities[i])

    return X, y

def encode_and_pad(prefixes, labels, max_prefix_length=15):
    """
    Encode activities as integers and pad prefixes.
    """

    all_activities = [a for p in prefixes for a in p] + labels

    encoder = LabelEncoder()
    encoder.fit(all_activities)

    encoded_X = []
    for p in prefixes:
        encoded = encoder.transform(p).tolist()
        padded = [0] * (max_prefix_length - len(encoded)) + encoded
        encoded_X.append(padded)

    encoded_y = encoder.transform(labels)

    return encoded_X, encoded_y


def run_preprocessing(input_csv):
    """
    End-to-end preprocessing pipeline:
    raw CSV -> processed CSVs
    """

    print("Loading data...")
    df = pd.read_csv(input_csv)
    print("Raw data loaded:", df.shape)

    print("Generating prefixes...")
    prefixes, labels = generate_prefixes(df)

    print("Encoding and padding...")
    X, y = encode_and_pad(prefixes, labels)

    print("Saving processed files...")
    pd.DataFrame(X).to_csv("data/processed/sequences.csv", index=False)
    pd.DataFrame(y, columns=["label"]).to_csv("data/processed/labels.csv", index=False)

    print("Preprocessing complete.")
    print("Total samples:", len(X))