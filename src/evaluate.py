import torch

def evaluate_model(model, X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        if isinstance(outputs, tuple):  # attention model
            outputs = outputs[0]

        preds = outputs.argmax(dim=1)
        correct = (preds == y_tensor).sum().item()

    accuracy = correct / len(y)
    return accuracy
