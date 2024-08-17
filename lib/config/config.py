from omegaconf import OmegaConf
import os
import hydra
cfg = OmegaConf.create()
def parse_cfg(cfg):
    if cfg.num_latent_code == -1:
        cfg.num_latent_code = cfg.num_train_frame
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.gpus}'
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)

@hydra.main(config_path='/home/wzx2021/gaussian-human/config', config_name='default')
def load_cfg(config):
    for item in config:
        if(item != 'dataset'):
            cfg.update({item : config[item]})
    cfg.merge_with(config.subject)
    parse_cfg(cfg)
load_cfg()