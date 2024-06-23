import yaml
from torch.optim import Adam, SGD


################# private functions ##################
### parse optimizer name to object
def __parse_optimizer(optimizer_name="Adam"):
    if optimizer_name == "Adam":
        return Adam
    elif optimizer_name == "SGD":
        return SGD
    else:
        print(f"Not support the following optimizer: {optimizer_name}")
        exit(1)

### read the whole config, and return the corresponding dict
def __load_config(path):
    with open(path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

################# public functions ##################
### extract only dip parameters and construct to corresponding python object
def load_dip_config(path):
    configs = __load_config(path)
    dip_config = configs['dip']
    dip_config["seed"] = configs["reproduce"]["seed"]
    dip_config["optimizer"] = __parse_optimizer(dip_config["optimizer"])
    return dip_config

def load_img_config(path):
    configs = __load_config(path)
    return configs["image"]["height"], configs["image"]["width"] 

def load_ddpm_config(path):
    configs = __load_config(path)
    ddpm_config = configs['ddpm']
    ddpm_config["seed"] = configs["reproduce"]["seed"]
    return ddpm_config

def load_guide_dip_config(path):
    configs = __load_config(path)
    guide_dip_config = configs['guide-dip']
    guide_dip_config["seed"] = configs["reproduce"]["seed"]
    return guide_dip_config

