import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from src.model_lstm import LSTMNextActivity
from src.model_attention import AttentionLSTM


def train_attention_model(X_train, y_train, vocab_size, epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AttentionLSTM(vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        acc = correct / total
        print(
            f"[ATTN] Epoch {epoch+1}/{epochs} | "
            f"Loss: {total_loss:.4f} | Acc: {acc:.4f}"
        )

    return model


def train_lstm(X_train, y_train, vocab_size, epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMNextActivity(vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        model.train()

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        acc = correct / total
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {total_loss:.4f} | "
            f"Accuracy: {acc:.4f}"
        )

    return model
