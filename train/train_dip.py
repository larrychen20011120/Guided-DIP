import os
import sys
import torch 
from torch.autograd import Variable
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()
DATA_ENTRY = str(os.getenv("DATA_ENTRY"))

# in order to import other files
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.model import DIP
from utils.config_utils import load_dip_config, load_img_config
from utils.image_utils import pillow2image, load_image_to_tensor, compute_psnr_score
from utils.plot_utils import plot_snapshots

def train_dip(model, target_image, clean_image=None, **config):

    # load parameters
    lr = config["learning_rate"]
    device = torch.device("cuda") if config["device"] == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    num_steps = config["num_steps"]


    losses, snapshots = [], []
    psnrs = []

    if clean_image is not None:
        clean_np_image = pillow2image(clean_image[0].permute(1,2,0))

    model = model.to(device)
    optimizer = config["optimizer"](model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    # sample a noise
    noise = Variable(torch.randn(target_image.shape))
    noise = noise.to(device)


    for _ in tqdm(range(num_steps)):
        # start training
        model.train()

        optimizer.zero_grad()
        output = model(noise)
        loss = mse(target_image, output)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()


        model.eval()
        with torch.no_grad():
            output = model(noise).cpu().detach()[0].permute(1,2,0)
            image = pillow2image(output)
            snapshots.append( image )

        if clean_image is not None:
            psnrs.append(
                compute_psnr_score(
                    clean_np_image, 
                    image
                )
            )

            

            # store the snapshot
            
        
    return losses, snapshots, psnrs


if __name__ == "__main__":

    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    config = load_dip_config(config_path)
    height, width = load_img_config(config_path)
    image_path = os.path.join(DATA_ENTRY, "Image_1.jpg")

    target_image = load_image_to_tensor(image_path, height, width)
    model = DIP(config["backbone_name"])
    print(model)

    snapshots = train_dip(model, target_image, **config)
    plot_snapshots(snapshots, plot_method="store", store_dir="assets", filename="test_for_dip")