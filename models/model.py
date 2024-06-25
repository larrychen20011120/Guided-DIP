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
from models.backbone import UNet, SegNet

def cosine_schedule(num_timesteps, s=0.008):
    def f(t):
        return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
    x = torch.linspace(0, num_timesteps, num_timesteps + 1)
    alphas_cumprod = f(x) / f(torch.tensor([0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clip(betas, 0.0001, 0.999)
    return betas

def linear_schedule(beta_start, beta_end, num_timesteps):
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    return betas

class DIP(nn.Module):
    def __init__(self, backbone_name):
        super(DIP, self).__init__()
        if backbone_name == "unet":
            self.backbone = UNet()
        elif backbone_name == "segnet":
            self.backbone = SegNet()
        else:
            print(f"You can use th backbone {backbone_name} !!!")
            exit(1)
        
    def forward(self, x):
        return self.backbone(x)

class DDPM:
    def __init__(self, **config):
        super(DDPM, self).__init__()

        if config["scheduler"] == "linear":
            self.betas = linear_schedule(config["beta_start"], config["beta_end"], config["num_time_steps"])
        elif config["scheduler"] == "cosine":
            self.betas = cosine_schedule(config["num_time_steps"])
            
        else:
            print(f"Do not support the scheduler {config['scheduler']}")
            exit(1)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x_start, t):
        noise = torch.randn_like(x_start)
        # normalize gaussian to (0, 1)
        noise = noise * 0.5 + 0.5  
        alpha_bar_t = self.alpha_bars[t]
        # weighted sum of noise the original image
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
    t = 100 
    
    print(f"Start adding noise to the target image in {image_path}...")
    noisy_image, _ = diffusion.forward_diffusion(target_image, t)
    noisy_image = noisy_image.squeeze(0)
    noisy_image = noisy_image.permute(1,2,0)
    
    plot_single_image(pillow2image(noisy_image), plot_method="store", filename="diffusion_example")
    print(f"The noisy version of image is store in assets")