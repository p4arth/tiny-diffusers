import torch
from torch import nn
from model.embedding import Embed

class DiffusionNetwork(nn.Module):
    def __init__(self, img_h = 28, img_w = 28):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.in_feats = img_h * img_w
        self.out_feats = img_h * img_w
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
            nn.Linear(in_features = 2048, out_features = 1024),
            nn.SiLU(),
            nn.Linear(in_features = 1024, out_features = 512),
            nn.SiLU(),
            nn.Linear(in_features = 512, out_features = self.out_feats),
            nn.SiLU()
        )

    def forward(self, x):
        return self.lin(x)

class Diffuser(nn.Module):
    def __init__(self, img_h, img_w, max_denoise_steps):
        super().__init__()
        self.time_embed = Embed(max_denoise_steps = max_denoise_steps, in_feats = (img_h * img_w))
        self.diffuser = DiffusionNetwork(img_h = img_h, img_w = img_w)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x, t):
        bs, c, h, w = x.shape
        x = self.diffuser(x.view(bs, c, h * w) + self.time_embed(t))
        return x.view(bs, c, h, w)