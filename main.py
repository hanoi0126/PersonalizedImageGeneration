import hydra
from omegaconf import OmegaConf

from fastcomposer.config import InferenceConfig

@hydra.main(version_base="1.1", config_path="configs/", config_name="config")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()