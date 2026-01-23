import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size + 1,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        # Attention layers
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)

        # Attention scores
        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(
            lstm_out * attn_weights.unsqueeze(-1),
            dim=1
        )
        logits = self.fc(context)

        return logits, attn_weights
