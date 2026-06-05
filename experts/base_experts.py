import torch.nn as nn


class BaseExpert(nn.Module):
    """
    All expert models must implement this structure so the
    Orchestrator can call .predict() on any of them.
    """

    def __init__(self):
        super(BaseExpert, self).__init__()

    def predict(self, x):
        # Every expert must have a predict method
        raise NotImplementedError("Each expert must implement its own predict() logic.")
