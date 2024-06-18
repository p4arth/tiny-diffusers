import torch
import matplotlib.pyplot as plt
import numpy as np

def sample(model,
           sigma = 1,
           steps = 1999, 
           out_channels = 28):
    in_feats = (1, 1, out_channels, out_channels)
    x0 = torch.rand_like(torch.zeros(in_feats))
    time_steps = list(reversed(list(range(steps))))
    model.eval()
    with torch.no_grad():
        for step in time_steps:
            eta = np.float32(np.random.normal(0, (sigma ** 2) * (step / steps), in_feats))
            x0 = model(x0, torch.LongTensor([step])) + eta
    return x0.squeeze(0)

def visualize_digit(digit_arr):
    plt.imshow(digit_arr.squeeze(0), cmap = 'gray')
    plt.show()