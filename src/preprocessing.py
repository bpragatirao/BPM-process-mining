import pandas as pd


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

        # Generate prefixes
        for i in range(1, len(activities)):
            prefix = activities[:i]
            if len(prefix) > max_prefix_length:
                prefix = prefix[-max_prefix_length:]

            X.append(prefix)
            y.append(activities[i])

    return X, y
