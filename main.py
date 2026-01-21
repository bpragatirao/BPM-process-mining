import pm4py
from src.data_loader import load_event_log
from src.preprocessing import generate_prefixes
from src.dataset import build_activity_vocab, encode_sequences

# Load
df = load_event_log("data/raw/bpi2017.xes")

# Preprocess
X, y = generate_prefixes(
    df,
    max_cases=3000,
    max_prefix_length=15
)

# Encode
activity_to_id, id_to_activity = build_activity_vocab(X, y)
X_encoded, y_encoded = encode_sequences(
    X, y,
    activity_to_id,
    max_prefix_length=15
)


print("Number of samples:", len(X))
print("First prefix:", X[0])
print("First label:", y[0])
print(X_encoded.shape, len(y_encoded))
