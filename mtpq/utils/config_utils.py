import yaml
from easydict import EasyDict

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(config)