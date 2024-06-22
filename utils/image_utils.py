from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_image_to_tensor(image_path, height, width):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((height, width))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image)
    # load image to batch_size `1`
    image = image.unsqueeze(0)
    return image

def pillow2image(image):
    image[image < 0] = 0
    image[image > 1] = 1
    image = image * 255
    return np.array( image, np.uint8 )

def compute_psnr_score(label, pred):
    return peak_signal_noise_ratio(label, pred)

def compute_ssim_score(label, pred):
    return structural_similarity(label, pred)