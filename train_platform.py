import torch
import torch.nn as nn
import torch.optim as optim
import csv
import json
import os
import re
from gateway import GatewayRouter
from experts.service_triage import TriageExpert
from experts.service_aml import AMLExpert
from experts.service_sentiment import SentimentExpert


def clean_and_tokenize(text):
    # Standard text normalization used in production NLP
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Strip punctuation, symbols, numbers
    return text.split()


def load_csv_dataset():
    dataset = []
    csv_path = os.path.join("data", "training_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Core dataset missing at {csv_path}. Run fetch_real_data.py first!")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append((row["text"], int(row["label"])))
    return dataset


def run_production_training():
    print("🚀 Initializing Real-World Dataset Tokenization and Preprocessing Pipeline...")
    raw_dataset = load_csv_dataset()

    # 1. Build an organic, data-driven vocabulary mapping dictionary
    vocab = {"<PAD>": 0, "<UNK>": 1}  # 0 is padding, 1 handles unknown words in production
    for text, _ in raw_dataset:
        words = clean_and_tokenize(text)
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    vocab_size = len(vocab)
    print(f"📊 Vocabulary Compiled. Total Unique Structural Words: {vocab_size}")

    # 2. Vectorize the entire text block into numerical matrices (Tensors)
    padded_inputs = []
    routing_targets = []
    for text, label in raw_dataset:
        words = clean_and_tokenize(text)
        tokens = [vocab.get(w, 1) for w in words]  # Fallback to <UNK> token if word is foreign
        padded_tokens = (tokens + [0] * 8)[:8]  # Maintain strict standardized shape of 8
        padded_inputs.append(padded_tokens)
        routing_targets.append(label)

    X = torch.tensor(padded_inputs)
    Y_router = torch.tensor(routing_targets)

    # 3. Train Gateway Router (Multi-class Cross-Entropy Evaluation)
    print("\n⚡ Optimizing Gateway Router Weights (Cross-Entropy Distribution)...")
    gateway = GatewayRouter(vocab_size)
    gateway_optimizer = optim.Adam(gateway.parameters(), lr=0.01)
    gateway_criterion = nn.CrossEntropyLoss()

    for epoch in range(120):
        gateway_optimizer.zero_grad()
        predictions = gateway(X)
        loss = gateway_criterion(predictions, Y_router)
        loss.backward()
        gateway_optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:02d}/120 | Cross-Entropy Routing Loss: {loss.item():.4f}")

    torch.save(gateway.state_dict(), "gateway.pth")
    print("✅ Gateway weights successfully optimized and saved to gateway.pth")

    # 4. Train Specialized Deep Learning Expert LSTM Architectures
    experts = {
        "triage": (TriageExpert(vocab_size), torch.tensor([[1.0] if y == 0 else [0.0] for y in Y_router])),
        "aml": (AMLExpert(vocab_size), torch.tensor([[1.0] if y == 1 else [0.0] for y in Y_router])),
        "sentiment": (SentimentExpert(vocab_size), torch.tensor([[1.0] if y == 2 else [0.0] for y in Y_router]))
    }

    print("\n🧠 Backpropagating Semantics Through Specialist LSTM Gates...")
    for name, (model, Y_expert) in experts.items():
        expert_optimizer = optim.Adam(model.parameters(), lr=0.005)
        expert_criterion = nn.BCEWithLogitsLoss()

        for epoch in range(80):
            expert_optimizer.zero_grad()
            outputs = model(X)
            loss = expert_criterion(outputs, Y_expert)
            loss.backward()
            expert_optimizer.step()

        weight_path = os.path.join("experts", f"service_{name}.pth")
        torch.save(model.state_dict(), weight_path)
        print(f"✅ Neural Weights Exported: {name.upper()} -> {weight_path}")


if __name__ == "__main__":
    run_production_training()