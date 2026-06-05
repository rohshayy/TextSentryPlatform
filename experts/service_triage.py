import torch
import torch.nn as nn
from experts.base_experts import BaseExpert


class TriageExpert(BaseExpert):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(TriageExpert, self).__init__()

        # Architecture: Embedding -> LSTM -> Linear (The Week 14 Syllabus)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)  # Outputs a binary Triage score

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        # Extract the last hidden state for classification
        h_n = hn.squeeze(0)
        return self.classifier(h_n)

    def predict(self, x):
        """
        Implementation of the BaseExpert interface.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
