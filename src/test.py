import torch
import sys, os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.diffuser import Diffuser
from utils.utils import visualize_digit, sample

parser = argparse.ArgumentParser(description='Denoising diffusion model testing')
parser.add_argument('--ckpt_path', help='Load pretrained model weights')
parser.add_argument('--max_denoise_steps', type=int, default=2000, help='Maximum number of denoising steps')
parser.add_argument('--sigma', type=float, default=1.0, help='Sigma value for Gaussian noise')

if __name__ == "__main__":
    args = vars(parser.parse_args())
    ckpt_path = args.get("ckpt_path")
    max_denoise_steps = args.get("max_denoise_steps", 2000)
    sigma = args.get("sigma", 1.0)

    model = Diffuser(img_h = 28, img_w = 28, max_denoise_steps=max_denoise_steps)
    state_dict = torch.load(ckpt_path)
    print("Best Epoch :", state_dict["epoch"])
    model.load_state_dict(state_dict["model"])
    visualize_digit(sample(model, steps = max_denoise_steps, sigma = sigma))