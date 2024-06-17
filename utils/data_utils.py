from PIL import Image
from torchvision import transforms
import numpy as np

def load_image_to_tensor(image_path, height, width):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((height, width))
    image = transforms.ToTensor()(image)
    # load image to batch_size `1`
    image = image.unsqueeze(0)
    return image

def pillow2image(image):
    image = image * 255
    return np.array( image, np.uint8 )


