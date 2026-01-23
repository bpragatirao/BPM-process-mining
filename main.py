import os
import torch

from src.data_loader import load_event_log
from src.preprocessing import generate_prefixes
from src.dataset import build_activity_vocab, encode_sequences
from src.train import train_lstm, train_attention_model
from src.evaluate import (
    evaluate_model,
    save_training_summary,
    save_model_comparison
)
from src.visualize_attention import visualize_attention


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "BPI Challenges 2017.xes")

# --------------------------------------------------
# 1. Load event log
# --------------------------------------------------
df = load_event_log(DATA_PATH)

# --------------------------------------------------
# 2. Generate prefixes (RAW STRINGS)
# --------------------------------------------------
X_raw, y_raw = generate_prefixes(
    df,
    max_cases=1000,
    max_prefix_length=15
)

print("Number of samples:", len(X_raw))
print("First prefix:", X_raw[0])
print("First label:", y_raw[0])

# --------------------------------------------------
# 3. Build vocabulary + encode
# --------------------------------------------------
activity_to_id, id_to_activity = build_activity_vocab(X_raw, y_raw)

X_encoded, y_encoded = encode_sequences(
    X_raw,
    y_raw,
    activity_to_id,
    max_prefix_length=15
)

# --------------------------------------------------
# 4. Train baseline LSTM
# --------------------------------------------------
baseline_model = train_lstm(
    X_encoded,
    y_encoded,
    vocab_size=len(activity_to_id),
    epochs=5
)

torch.save(
    baseline_model.state_dict(),
    "outputs/models/baseline_lstm/model.pt"
)

# --------------------------------------------------
# 5. Evaluate baseline LSTM (ENCODED DATA ONLY)
# --------------------------------------------------
baseline_acc, baseline_loss = evaluate_model(
    baseline_model,
    X_encoded,
    y_encoded
)

save_training_summary(
    model_name="baseline_lstm",
    accuracy=baseline_acc,
    loss=baseline_loss,
    num_samples=len(y_encoded),
    output_dir="outputs/models/baseline_lstm"
)

# --------------------------------------------------
# 6. Train attention LSTM
# --------------------------------------------------
attention_model = train_attention_model(
    X_encoded,
    y_encoded,
    vocab_size=len(activity_to_id),
    epochs=5
)

torch.save(
    attention_model.state_dict(),
    "outputs/models/attention_lstm/model.pt"
)

# --------------------------------------------------
# 7. Evaluate attention LSTM
# --------------------------------------------------
attention_acc, attention_loss = evaluate_model(
    attention_model,
    X_encoded,
    y_encoded
)

save_training_summary(
    model_name="attention_lstm",
    accuracy=attention_acc,
    loss=attention_loss,
    num_samples=len(y_encoded),
    output_dir="outputs/models/attention_lstm"
)

# --------------------------------------------------
# 8. Save model comparison
# --------------------------------------------------
baseline_results = {
    "accuracy": baseline_acc,
    "loss": baseline_loss
}

attention_results = {
    "accuracy": attention_acc,
    "loss": attention_loss
}

save_model_comparison(
    baseline_results,
    attention_results,
    output_dir="outputs/models"
)

# --------------------------------------------------
# 9. Visualize attention (RAW STRINGS ONLY)
# --------------------------------------------------
sample_prefix = X_raw[0]

visualize_attention(
    attention_model,
    sample_prefix,
    activity_to_id,
    id_to_activity
)

# --------------------------------------------------
# 10. Final output
# --------------------------------------------------
print("\nModel Comparison")
print("----------------")
print(f"LSTM Accuracy      : {baseline_acc:.4f}")
print(f"Attention Accuracy : {attention_acc:.4f}")
