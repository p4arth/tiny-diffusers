import torch
import sys, os
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.diffuser import Diffuser

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
# if torch.backends.mps.is_available():
#     DEVICE = "mps"
print(f"Running code on device : {DEVICE}")

def train(data, model, criterion, optimizer, denoise_steps, epochs, starting_epoch, sigma):
    model.train()
    min_loss = float('inf')
    for epoch in range(int(starting_epoch) + 1, epochs):
        pbar = tqdm(enumerate(data), desc = f"Epoch[{epoch}]", total=(len(data)))
        losses = torch.zeros(len(data)).to(DEVICE)
        for idx, [x0, _] in pbar:
            x0 = x0.to(DEVICE)
            optimizer.zero_grad()
            t = torch.randint(0, denoise_steps, (1,))
            noise_xt = torch.tensor(
                np.float32(np.random.normal(0, (sigma ** 2) * (t / (denoise_steps - 1)), x0.shape)), 
                device=DEVICE
            )
            noise_xt_delta = torch.tensor(
                np.float32(np.random.normal(0, (sigma ** 2) * (((t + 1) / (denoise_steps - 1))), x0.shape)),
                device = DEVICE
            )
            xt = x0 + noise_xt
            xt_delta = xt + noise_xt_delta
            xt_hat = model(xt_delta, t)
            loss = criterion(xt_hat, xt)
            loss.backward()
            optimizer.step()
            losses[idx] = loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        mean_loss = torch.mean(losses).item()
        print(f"Average Epoch Loss = {mean_loss:.4f}")
        if min_loss > mean_loss:
            min_loss = mean_loss
            torch.save(
                {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                 "epoch": epoch}, 
                f"pretrained/best_loss.pth"
            )
        torch.save(
            {"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch}, 
            f"pretrained/latest_model.pth"
        )


if __name__ == "__main__":
    load_pretrained = False
    batch_size = 256
    max_denoise_steps = 2000
    epochs = 5000
    lr = 0.0002
    sigma =  1
    
    x = DataLoader(MNIST(root = ".", download = True, transform = ToTensor()), batch_size = batch_size, shuffle=True)
    model = Diffuser(img_h = 28, img_w = 28, max_denoise_steps=max_denoise_steps).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = lr)
    
    os.makedirs("pretrained", exist_ok=True)
    if load_pretrained:
        name = f"pretrained/latest_model.pth"
        state_dict = torch.load(name)
        epoch_run = state_dict["epoch"]
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
    train(
        data = x,
        model = model,
        criterion = criterion,
        optimizer = optimizer,
        denoise_steps = max_denoise_steps,
        epochs = epochs,
        starting_epoch = epoch_run if load_pretrained else 0,
        sigma = sigma
    )