import os
import types
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from omegaconf import DictConfig
from transformers import CLIPTokenizer

from fastcomposer.data import DemoDataset
from fastcomposer.model import FastComposerModel
from fastcomposer.pipeline import (
    stable_diffusion_call_with_references_delayed_conditioning,
)
from fastcomposer.transforms import get_object_transforms


@hydra.main(version_base=None, config_path="../configs", config_name="infer_config")
@torch.no_grad()
def main(cfg: DictConfig) -> None:

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.pretrained_model_path, torch_dtype=weight_dtype
    )

    model = FastComposerModel.from_pretrained(cfg)  # TODO: Check this

    ckpt_name = "pytorch_model.bin"

    model.load_state_dict(
        torch.load(Path(cfg.finetuned_model_path) / ckpt_name, map_location="cpu")
    )

    model = model.to(device=accelerator.device, dtype=weight_dtype)

    pipe.unet = model.unet

    if cfg.enable_xformers_memory_efficient_attention:
        pipe.unet.enable_xformers_memory_efficient_attention()

    pipe.text_encoder = model.text_encoder
    pipe.image_encoder = model.image_encoder

    pipe.postfuse_module = model.postfuse_module

    pipe.inference = types.MethodType(
        stable_diffusion_call_with_references_delayed_conditioning, pipe
    )

    del model

    pipe = pipe.to(accelerator.device)

    # Set up the dataset
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_path,
        subfolder="tokenizer",
        revision=cfg.revision,
    )

    object_transforms = get_object_transforms(cfg)  # TODO: Check this

    demo_dataset = DemoDataset(
        test_caption=cfg.caption,
        test_reference_folder=cfg.reference_dir,
        tokenizer=tokenizer,
        object_transforms=object_transforms,
        device=accelerator.device,
        max_num_objects=cfg.max_num_objects,
    )

    image_ids = os.listdir(cfg.reference_dir)
    print(f"Image IDs: {image_ids}")
    demo_dataset.set_image_ids(image_ids)

    unique_token = "<|image|>"

    prompt = cfg.caption
    prompt_text_only = prompt.replace(unique_token, "")

    os.makedirs(cfg.output_dir, exist_ok=True)

    batch = demo_dataset.get_data()

    input_ids = batch["input_ids"].to(accelerator.device)
    # text = tokenizer.batch_decode(input_ids)[0]
    print(prompt)
    # print(input_ids)
    image_token_mask = batch["image_token_mask"].to(accelerator.device)

    # print(image_token_mask)
    all_object_pixel_values = (
        batch["object_pixel_values"].unsqueeze(0).to(accelerator.device)
    )
    num_objects = batch["num_objects"].unsqueeze(0).to(accelerator.device)

    all_object_pixel_values = all_object_pixel_values.to(
        dtype=weight_dtype, device=accelerator.device
    )

    object_pixel_values = all_object_pixel_values  # [:, 0, :, :, :]
    if pipe.image_encoder is not None:
        object_embeds = pipe.image_encoder(object_pixel_values)
    else:
        object_embeds = None

    encoder_hidden_states = pipe.text_encoder(
        input_ids, image_token_mask, object_embeds, num_objects
    )[0]

    encoder_hidden_states_text_only = pipe._encode_prompt(
        prompt_text_only,
        accelerator.device,
        cfg.num_images_per_prompt,
        do_classifier_free_guidance=False,
    )

    encoder_hidden_states = pipe.postfuse_module(
        encoder_hidden_states,
        object_embeds,
        image_token_mask,
        num_objects,
    )

    cross_attention_kwargs = {}

    images = pipe.inference(
        prompt_embeds=encoder_hidden_states,
        num_inference_steps=cfg.inference_steps,
        height=cfg.generate_height,
        width=cfg.generate_width,
        guidance_scale=cfg.guidance_scale,
        num_images_per_prompt=cfg.num_images_per_prompt,
        cross_attention_kwargs=cross_attention_kwargs,
        prompt_embeds_text_only=encoder_hidden_states_text_only,
        start_merge_step=cfg.start_merge_step,
    ).images

    for instance_id in range(cfg.num_images_per_prompt):
        images[instance_id].save(
            os.path.join(
                cfg.output_dir,
                f"output_{instance_id}.png",
            )
        )


if __name__ == "__main__":
    main()
