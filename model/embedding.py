import torch
from torch import nn

class Embed(nn.Module):
    def __init__(self, max_denoise_steps, in_feats):
        super().__init__()
        self.embed = torch.nn.Embedding(
            num_embeddings = max_denoise_steps,
            embedding_dim = in_feats
        )
    def forward(self, x):
        return self.embed(x)