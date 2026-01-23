from src.data_loader import load_event_log
from src.preprocessing import generate_prefixes
from src.dataset import build_activity_vocab, encode_sequences
from src.train import train_lstm
from src.visualize_attention import visualize_attention
from src.evaluate import evaluate_model
from src.train import train_attention_model
import os

# Absolute-safe path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "BPI Challenges 2017.xes")

# Load data
df = load_event_log(DATA_PATH)

# Preprocess
X, y = generate_prefixes(
    df,
    max_cases=1000,
    max_prefix_length=15
)

print("Number of samples:", len(X))
print("First prefix:", X[0])
print("First label:", y[0])

# Encode
activity_to_id, id_to_activity = build_activity_vocab(X, y)
X_encoded, y_encoded = encode_sequences(
    X, y, activity_to_id, max_prefix_length=15
)

# Train
model = train_lstm(
    X_encoded,
    y_encoded,
    vocab_size=len(activity_to_id),
    epochs=5
)

attn_model = train_attention_model(
    X_encoded,
    y_encoded,
    vocab_size=len(activity_to_id),
    epochs=5
)

sample_prefix = X[0]

visualize_attention(
    attn_model,
    sample_prefix,
    activity_to_id,
    id_to_activity
)

lstm_acc = evaluate_model(model, X_encoded, y_encoded)
attn_acc = evaluate_model(attn_model, X_encoded, y_encoded)

print("\nModel Comparison")
print("----------------")
print(f"LSTM Accuracy      : {lstm_acc:.4f}")
print(f"Attention Accuracy : {attn_acc:.4f}")

