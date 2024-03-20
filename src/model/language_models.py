import torch
import torch.nn as nn
from torch.nn import functional as fn

from src.model.hyper_paramters import SEED_NUMBER

torch.manual_seed(SEED_NUMBER)


class BigramLanguageMode(nn.Module):
    def __init__(self, vocabulary_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)  # (BTC)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = fn.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, maximum_new_tokens):
        for _ in range(maximum_new_tokens):
            """Get predictions"""
            logits, loss = self(index)
            """Only for last time step"""
            logits = logits[:, -1, :]  # logtis in (B, C)
            """Get probabilities"""
            probabilities = fn.softmax(logits, dim=1)  # probabilities in (B, C)
            """Sample from distribution"""
            index_for_next = torch.multinomial(probabilities, num_samples=1)
            """Append sampled index to run sequence"""
            index = torch.cat((index, index_for_next), dim=1)  # (B, T +1)

        return index
