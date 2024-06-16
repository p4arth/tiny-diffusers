import torch
from diffuser import Diffuser
from utils import visualize_digit
from utils import sample

beta = 0.001
model = Diffuser(beta = beta)
state_dict = torch.load(f"pretrained/best_loss.pth")
print("Best Epoch :", state_dict["epoch"])
model.load_state_dict(state_dict["model"])
visualize_digit(sample(model, steps = 2000, beta = beta))