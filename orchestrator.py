import json
import torch
import os
import re
from gateway import GatewayRouter
from experts.service_triage import TriageExpert
from experts.service_aml import AMLExpert
from experts.service_sentiment import SentimentExpert


class PlatformOrchestrator:
    def __init__(self, config_path="config.json", vocab_path="vocab.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        vocab_size = len(self.vocab)

        # 1. Initialize and Load Trained Gateway Router Intelligence
        self.gateway = GatewayRouter(vocab_size)
        if os.path.exists("gateway.pth"):
            self.gateway.load_state_dict(torch.load("gateway.pth", weights_only=True))
            self.gateway.eval()
            print("✅ Loaded Gateway Router intelligence mapping.")
        else:
            print("⚠️ Warning: gateway.pth missing. Run train_platform.py first.")

        # 2. Initialize and Load Expert LSTM Models
        self.experts = {
            "triage": TriageExpert(vocab_size),
            "aml": AMLExpert(vocab_size),
            "sentiment": SentimentExpert(vocab_size)
        }

        for name, expert in self.experts.items():
            path = self.config["experts"][name]
            if os.path.exists(path):
                expert.load_state_dict(torch.load(path, weights_only=True))
                expert.eval()
                print(f"✅ Loaded {name.upper()} neural network architectures.")
            else:
                print(f"❌ Error: Weights missing for {name} at {path}")

    def process_query(self, text):
        # Clean text exactly how it was cleaned during model training
        text_clean = text.lower()
        text_clean = re.sub(r'[^a-z\s]', '', text_clean)
        words = text_clean.split()

        # Map words to IDs, default to 1 (<UNK>) if word is unseen
        tokens = [self.vocab.get(w, 1) for w in words]
        padded_tokens = (tokens + [0] * 8)[:8]
        text_tensor = torch.tensor([padded_tokens])

        # Run live non-gradient tracking forward inference
        with torch.no_grad():
            gateway_output = self.gateway(text_tensor)
            route_idx = self.gateway.get_route(gateway_output)
            route_name = self.config["labels"][route_idx]

            expert = self.experts[route_name]
            result = expert.predict(text_tensor)

        return route_name, result.item()


if __name__ == "__main__":
    orchestrator = PlatformOrchestrator()

    # 🧪 LIVE PORTFOLIO TEST CASES
    # Swap these strings out to verify true semantic model switching!
    test_inputs = [
        "Database error connection timeout occurred on server",
        "i need an apple",
        "The application interface update was terrible and very slow"
    ]

    print("\n" + "=" * 50)
    print("      EXECUTING LIVE SYSTEM INTERPRETATION RUN      ")
    print("=" * 50)

    for sentence in test_inputs:
        route, confidence = orchestrator.process_query(sentence)
        print(f"Input Text: '{sentence}'")
        print(f"Routed To : {route.upper()}")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 50)