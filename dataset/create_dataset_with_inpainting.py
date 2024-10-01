import hydra
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="inpainting")
def inpainting(cfg: DictConfig) -> None:
    pipeline = AutoPipelineForInpainting.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16
    )
    pipeline.enable_model_cpu_offload()

    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    pipeline.enable_xformers_memory_efficient_attention()

    init_image = load_image(cfg.init_image)
    mask_image = load_image(cfg.mask_image)

    image = pipeline(
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        image=init_image,
        mask_image=mask_image,
    ).images[0]
    images = make_image_grid([init_image, mask_image, image], rows=1, cols=3)

    # save image
    image.save(f"{cfg.output_dir}/image.png")
    images.save(f"{cfg.output_dir}/result_grid.png")


if __name__ == "__main__":
    inpainting()
