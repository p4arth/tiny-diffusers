import torch
import matplotlib.pyplot as plt

def sample(model,
           beta,
           steps = 10, 
           out_channels = 28):
    torch.manual_seed(42)
    out_feats = (1, 1, out_channels * out_channels)
    alpha = (1 - beta)
    xt = torch.rand_like(torch.zeros(out_feats))
    model.eval()
    with torch.no_grad():
        for step in range(steps, -1, -1):
            a_tilde = alpha**step
            z = torch.normal(xt) if step > 0 else 0
            e_theta = ((1 - alpha) / (((1 - a_tilde) ** 0.5) + 0.00001)) * model(xt, sampling = True)
            x_t_nxt = ((1 / (alpha ** 0.5)) * (xt - e_theta)) + (beta ** step) * z
    return x_t_nxt.view(1, out_channels, out_channels)

def visualize_digit(digit_arr):
    plt.imshow(digit_arr.squeeze(0).detach().cpu().numpy(), cmap = 'gray')
    plt.show()
# var_coeff = ((1 - alpha) * (1 - (alpha ** (step - 1)))) / (1 - a_tilde + 0.00001)