import torch
from torch import nn

class Diffuser(nn.Module):
    def __init__(self, beta, in_channels = 28, out_channels = 28, max_t = 2000, device = "cpu"):
        super().__init__()
        self.beta = beta
        self.alpha = 1 - beta
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_feats = in_channels * in_channels
        self.out_feats = out_channels * out_channels
        self.out_feats = 784
        self.max_denoise_steps = 2000
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.max_denoise_steps,
            embedding_dim=self.in_feats
        )
        self.lin = nn.Sequential(
            nn.Linear(in_features = self.in_feats, out_features = 512),
            nn.SiLU(),
            nn.Linear(in_features = 512, out_features = 1024),
            nn.SiLU(),
            nn.Linear(in_features = 1024, out_features = 2048),
            nn.SiLU(),
            nn.Linear(in_features = 2048, out_features = 2048),
            nn.SiLU(),
            nn.Linear(in_features = 2048, out_features = 2048),
            nn.SiLU(),
            nn.Linear(in_features = 2048, out_features = 2048),
            nn.SiLU(),
            nn.Linear(in_features = 2048, out_features = 2048),
            nn.SiLU(),
            nn.Linear(in_features = 2048, out_features = 2048),
            nn.SiLU(),
            nn.Linear(in_features = 2048, out_features = 2048),
            nn.SiLU(),
            nn.Linear(in_features = 2048, out_features = 2048),
            nn.SiLU(),
            nn.Linear(in_features = 2048, out_features = 1024),
            nn.SiLU(),
            nn.Linear(in_features = 1024, out_features = 512),
            nn.SiLU(),
            nn.Linear(in_features = 512, out_features = self.out_feats),
            nn.SiLU()
        )
        self.device = device
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x, t = None, sampling = False):
        if sampling:
            return self.lin(x)
        bs, c, h, w = x.shape

        # Noise -> ShapeOf(original_image)
        eps = torch.normal(0, 1, x.shape).to(self.device)

        # Coeff for noising
        alpha_t = (self.alpha ** t)

        # Forward Process
        x_noisy = ((alpha_t ** 0.5) * x) + (((1 - alpha_t) ** 0.5) * eps)
        
        # Time embedding
        time_embedding = self.embedding(torch.LongTensor([t])).to(self.device)

        # Into the model we go
        e_theta = self.lin(x_noisy.view(bs, c, h * w) + time_embedding)

        return eps, e_theta.view(bs, c, h, w)
