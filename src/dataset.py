from collections import Counter
import numpy as np


def build_activity_vocab(X, y):
    all_activities = []
    for prefix in X:
        all_activities.extend(prefix)
    all_activities.extend(y)

    activity_to_id = {
        act: idx + 1
        for idx, act in enumerate(set(all_activities))
    }
    id_to_activity = {v: k for k, v in activity_to_id.items()}

    return activity_to_id, id_to_activity


def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)

    for i, seq in enumerate(sequences):
        trunc = seq[-maxlen:]
        padded[i, -len(trunc):] = trunc

    return padded


def encode_sequences(X, y, activity_to_id, max_prefix_length):
    X_encoded = [
        [activity_to_id[a] for a in prefix]
        for prefix in X
    ]

    X_padded = pad_sequences(X_encoded, max_prefix_length)

    y_encoded = np.array([activity_to_id[label] for label in y])

    return X_padded, y_encoded
