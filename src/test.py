import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.diffuser import Diffuser
from utils.utils import visualize_digit, sample

beta = 0.001
model = Diffuser(img_h = 28, img_w = 28, max_denoise_steps=2000)
state_dict = torch.load(f"pretrained/best_loss.pth")
print("Best Epoch :", state_dict["epoch"])
model.load_state_dict(state_dict["model"])
visualize_digit(sample(model, steps = 2000))