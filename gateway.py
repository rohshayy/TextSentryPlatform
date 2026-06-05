import torch
import torch.nn as nn
import json

class GatewayRouter(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(GatewayRouter, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Simple classifier for routing: 3 outputs (triage, aml, sentiment)
        self.classifier = nn.Linear(embed_dim, 3)

    def forward(self, x):
        # Average the embeddings to get a fast "document context"
        embedded = self.embedding(x).mean(dim=1)
        return self.classifier(embedded)

    @staticmethod
    def get_route(prediction_tensor):
        # Returns the index of the highest probability expert
        return torch.argmax(prediction_tensor, dim=1).item()