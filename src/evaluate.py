import torch
import torch.nn as nn
import pandas as pd
import os

def evaluate_model(model, X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        outputs = model(X_tensor)
        if isinstance(outputs, tuple):  # attention model
            outputs = outputs[0]

        loss = criterion(outputs, y_tensor).item()
        preds = outputs.argmax(dim=1)
        correct = (preds == y_tensor).sum().item()

    accuracy = correct / len(y)
    return accuracy,loss


def save_training_summary(
    model_name,
    accuracy,
    loss,
    num_samples,
    output_dir
):
    summary = {
        "metric": [
            "accuracy",
            "loss",
            "num_samples"
        ],
        "value": [
            accuracy,
            loss,
            num_samples
        ]
    }

    df = pd.DataFrame(summary)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(
        f"{output_dir}/training_summary.csv",
        index=False
    )


def save_model_comparison(baseline_results, attention_results, output_dir="outputs/models"):
    """
    Save model comparison results as CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    comparison_df = pd.DataFrame([
        {
            "model": "Baseline LSTM",
            "accuracy": baseline_results["accuracy"],
            "loss": baseline_results["loss"]
        },
        {
            "model": "Attention LSTM",
            "accuracy": attention_results["accuracy"],
            "loss": attention_results["loss"]
        }
    ])

    comparison_df.to_csv(
        os.path.join(output_dir, "comparison.csv"),
        index=False
    )
