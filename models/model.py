import torch
import torch.nn as nn
import os
import sys
from dotenv import load_dotenv

load_dotenv()
DATA_ENTRY = str(os.getenv("DATA_ENTRY"))

# in order to import other files
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from models.backbone import UNet

class DIP(nn.Module):
    def __init__(self, **config):
        super(DIP, self).__init__()
        self.backbone = UNet(**config)
        
    def forward(self, x):
        return self.backbone(x)

class DDPM:
    def __init__(self, **config):
        super(DDPM, self).__init__()
        self.betas = torch.linspace(config["beta_start"], config["beta_end"], config["num_time_steps"])
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x_start, t):
        noise = torch.randn_like(x_start)
        noise = noise * 0.5 + 0.5  # Optional: Comment this line out to use standard noise
        alpha_bar_t = self.alpha_bars[t]
        x_t = torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise



from utils.config_utils import load_ddpm_config, load_img_config
from utils.image_utils import pillow2image, load_image_to_tensor
from utils.plot_utils import plot_single_image

if __name__ == "__main__":

    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    config = load_ddpm_config(config_path)
    height, width = load_img_config(config_path)
    image_path = os.path.join(DATA_ENTRY, "Image_1.jpg")

    target_image = load_image_to_tensor(image_path, height, width)
    diffusion = DDPM(**config)
    t = 100 #torch.tensor([100])
    
    print(f"Start adding noise to the target image in {image_path}...")
    noisy_image, _ = diffusion.forward_diffusion(target_image, t)
    noisy_image = noisy_image.squeeze(0)
    noisy_image = noisy_image.permute(1,2,0)
    
    plot_single_image(pillow2image(noisy_image), plot_method="store", filename="diffusion_example")
    print(f"The noisy version of image is store in assets")