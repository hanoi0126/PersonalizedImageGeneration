import logging
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import diffusers
import hydra
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

import wandb
from fastcomposer.data import FastComposerDataset, get_data_loader
from fastcomposer.model import FastComposerModel
from fastcomposer.transforms import (
    get_object_processor,
    get_object_transforms,
    get_train_transforms_with_segmap,
)

logger = get_logger(__name__)

wandb.init(
    project="personalization-training",
    entity="hiroto-weblab",
    name=datetime.now().strftime("%Y-%m-%d/%H-%M-%S"),
    # settings=wandb.Settings(mode="disabled"),
)


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def train(cfg: DictConfig) -> None:

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        logging_dir=cfg.logging_dir,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)
        if cfg.logging_dir is not None:
            os.makedirs(cfg.logging_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    t = time.localtime()
    str_m_d_y_h_m_s = time.strftime("%m-%d-%Y_%H-%M-%S", t)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=(
            [
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    os.path.join(cfg.logging_dir, f"{str_m_d_y_h_m_s}.log")
                ),
            ]
            if accelerator.is_main_process
            else []
        ),
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_path,
        subfolder="tokenizer",
        revision=cfg.revision,
    )

    model = FastComposerModel.from_pretrained(cfg)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # freeze all params in the model
    for param in model.parameters():
        param.requires_grad = False
        param.data = param.data.to(weight_dtype)

    if cfg.load_model is not None:
        model.load_state_dict(
            torch.load(Path(cfg.load_model) / "pytorch_model.bin", map_location="cpu")
        )

    model.unet.requires_grad_(True)
    model.unet.to(torch.float32)

    if cfg.text_image_linking in ["postfuse"] and not cfg.freeze_postfuse_module:
        model.postfuse_module.requires_grad_(True)
        model.postfuse_module.to(torch.float32)

    if cfg.train_text_encoder:
        model.text_encoder.requires_grad_(True)
        model.text_encoder.to(torch.float32)

    if cfg.train_image_encoder:
        if cfg.image_encoder_trainable_layers > 0:
            for idx in range(cfg.image_encoder_trainable_layers):
                model.image_encoder.vision_model.encoder.layers[
                    -1 - idx
                ].requires_grad_(True)
                model.image_encoder.vision_model.encoder.layers[-1 - idx].to(
                    torch.float32
                )
        else:
            model.image_encoder.requires_grad_(True)
            model.image_encoder.to(torch.float32)

    # Create EMA for the unet.
    if cfg.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_model_path,
            subfolder="unet",
            revision=cfg.revision,
        )
        model.load_ema(ema_unet)
        if cfg.load_model is not None:
            model.ema_param.load_state_dict(
                torch.load(
                    Path(cfg.load_model) / "custom_checkpoint_0.pkl",
                    map_location="cpu",
                )
            )

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.gradient_checkpointing:
        if cfg.train_text_encoder:
            model.text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate
            * cfg.gradient_accumulation_steps
            * cfg.train_batch_size
            * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    unet_params = list([p for p in model.unet.parameters() if p.requires_grad])
    other_params = list(
        [p for n, p in model.named_parameters() if p.requires_grad and "unet" not in n]
    )
    parameters = unet_params + other_params

    optimizer = optimizer_cls(
        [
            {"params": unet_params, "lr": cfg.learning_rate * cfg.unet_lr_scale},
            {"params": other_params, "lr": cfg.learning_rate},
        ],
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    train_transforms = get_train_transforms_with_segmap(cfg)
    object_transforms = get_object_transforms(cfg)
    object_processor = get_object_processor(cfg)

    if cfg.object_types is None or cfg.object_types == "all":
        object_types = None  # all object types
    else:
        object_types = cfg.object_types.split("_")
        print(f"Using object types: {object_types}")

    train_dataset = FastComposerDataset(
        cfg.dataset_name,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=accelerator.device,
        max_num_objects=cfg.max_num_objects,
        num_image_tokens=cfg.num_image_tokens,
        object_appear_prob=cfg.object_appear_prob,
        uncondition_prob=cfg.uncondition_prob,
        text_only_prob=cfg.text_only_prob,
        object_types=object_types,
        split="train",
        min_num_objects=cfg.min_num_objects,
        balance_num_objects=cfg.balance_num_objects,
    )

    train_dataloader = get_data_loader(train_dataset, cfg.train_batch_size)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.use_ema:
        accelerator.register_for_checkpointing(model.module.ema_param)
        model.module.ema_param.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        accelerator.init_trackers("FastComposer", config=config_dict)

    # Train!
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name} with shape {param.shape}")

    total_batch_size = (
        cfg.train_batch_size
        * accelerator.num_processes
        * cfg.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = cfg.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = cfg.global_step

            first_epoch = global_step // num_update_steps_per_epoch

            # move all the state to the correct device
            model.to(accelerator.device)
            if cfg.use_ema:
                model.module.ema_param.to(accelerator.device)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.num_train_epochs):
        model.train()
        train_loss = 0.0
        denoise_loss = 0.0
        localization_loss = 0.0
        face_separation_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            progress_bar.set_description("Global step: {}".format(global_step))

            with accelerator.accumulate(model), torch.backends.cuda.sdp_kernel(
                enable_flash=not cfg.disable_flashattention
            ):
                if step % 100 == 0:
                    return_dict = model(batch, noise_scheduler, return_image=True)
                else:
                    return_dict = model(batch, noise_scheduler)
                loss = return_dict["loss"]

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                avg_denoise_loss = accelerator.gather(
                    return_dict["denoise_loss"].repeat(cfg.train_batch_size)
                ).mean()
                denoise_loss += (
                    avg_denoise_loss.item() / cfg.gradient_accumulation_steps
                )

                if "localization_loss" in return_dict:
                    avg_localization_loss = accelerator.gather(
                        return_dict["localization_loss"].repeat(cfg.train_batch_size)
                    ).mean()
                    localization_loss += (
                        avg_localization_loss.item() / cfg.gradient_accumulation_steps
                    )

                if "face_separation_loss" in return_dict:
                    avg_face_separation_loss = accelerator.gather(
                        return_dict["face_separation_loss"].repeat(cfg.train_batch_size)
                    ).mean()
                    # 各値を float に変換してから計算
                    avg_face_separation_loss_value = float(
                        avg_face_separation_loss.item()
                    ) / float(cfg.gradient_accumulation_steps)
                    adjusted_loss = (
                        float(cfg.face_separation_delta)
                        - avg_face_separation_loss_value
                    )
                    # 条件分岐で adjusted_loss を加算
                    if adjusted_loss > 0.0:
                        face_separation_loss += adjusted_loss
                        logging.info(
                            f"face_separation_loss: {face_separation_loss}, adjusted_loss: {adjusted_loss}, in image: {batch['image_ids']}"
                        )

                # Adjust total loss depending on cfg.pass_face_separation_loss
                if cfg.pass_face_separation_loss:
                    loss = loss + face_separation_loss
                else:
                    loss = loss + torch.tensor(face_separation_loss).detach()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.use_ema:
                    model.module.ema_param.step(model.module.unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "denoise_loss": denoise_loss,
                        "localization_loss": localization_loss,
                        "face_separation_loss": face_separation_loss,
                    },
                    step=global_step,
                )

                # log generate image
                if step % 100 == 0:
                    accelerator.log(
                        {
                            "generated_images": [
                                wandb.Image(
                                    return_dict["decoded_ref_image"],
                                    caption=f"ref-step-{global_step:08}-batch-{step:08}",
                                ),
                                wandb.Image(
                                    return_dict["decoded_gen_image"],
                                    caption=f"gen-step-{global_step:08}-batch-{step:08}",
                                ),
                            ]
                        },
                        step=global_step,
                    )

                train_loss = 0.0
                denoise_loss = 0.0
                localization_loss = 0.0
                face_separation_loss = 0.0

                if (
                    global_step % cfg.checkpointing_steps == 0
                    and accelerator.is_local_main_process
                ):
                    save_path = os.path.join(
                        cfg.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    if cfg.keep_only_last_checkpoint:
                        # Remove all other checkpoints
                        for file in os.listdir(cfg.output_dir):
                            if file.startswith(
                                "checkpoint"
                            ) and file != os.path.basename(save_path):
                                ckpt_num = int(file.split("-")[1])
                                if (
                                    cfg.keep_interval is None
                                    or ckpt_num % cfg.keep_interval != 0
                                ):
                                    logger.info(f"Removing {file}")
                                    shutil.rmtree(os.path.join(cfg.output_dir, file))

            logs = {
                "l_noise": return_dict["denoise_loss"].detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            if "localization_loss" in return_dict:
                logs["l_loc"] = return_dict["localization_loss"].detach().item()

            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if cfg.use_ema:
            model.ema_param.copy_to(model.unet.parameters())

        pipeline = model.to_pipeline()
        pipeline.save_pretrained(cfg.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    train()
