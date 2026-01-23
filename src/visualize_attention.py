import torch
import matplotlib.pyplot as plt


def visualize_attention(
    model,
    sample_prefix,
    activity_to_id,
    id_to_activity
):
    model.eval()

    # Encode prefix
    encoded = [activity_to_id[a] for a in sample_prefix]
    x = torch.tensor(encoded).unsqueeze(0)

    with torch.no_grad():
        _, attn_weights = model(x)

    attn = attn_weights.squeeze().numpy()

    activities = sample_prefix[-len(attn):]

    # Print textual explanation
    print("\nAttention Weights:")
    for act, w in zip(activities, attn):
        print(f"{act:30s} -> {w:.3f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(attn)), attn)
    plt.xticks(range(len(attn)), activities, rotation=45, ha="right")
    plt.title("Attention over Process Prefix")
    plt.tight_layout()
    plt.show()
