import torch
import torch.nn as nn
from experts.base_experts import BaseExpert

class SentimentExpert(BaseExpert):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(SentimentExpert, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn.squeeze(0))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
