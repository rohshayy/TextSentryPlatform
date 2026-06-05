import torch
import json
import re

class RealTextProcessor:
    def __init__(self, vocab_path="vocab.json"):
        # Load the vocabulary created during training
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

    def text_to_tensor(self, text, seq_len=8):
        # 1. Clean the text
        clean = re.sub(r'[^a-z\s]', '', text.lower())
        # 2. Tokenize (Map words to IDs)
        tokens = [self.vocab.get(w, 0) for w in clean.split()]
        # 3. Pad to match the model's input size
        padded = (tokens + [0]*seq_len)[:seq_len]
        return torch.tensor([padded])
