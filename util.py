import mlflow
from omegaconf import OmegaConf

def log_params_from_omegaconf_dict(config):
    for key, value in OmegaConf.to_container(config, resolve=True).items():
        mlflow.log_param(key, value)