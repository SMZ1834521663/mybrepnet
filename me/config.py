from omegaconf import OmegaConf

def load_config(yaml_path='./me/config.yaml'):
    cfg = OmegaConf.load(yaml_path)
    return cfg

cfg=load_config()