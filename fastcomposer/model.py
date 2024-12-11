import gc
import os
import types
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from facenet_pytorch import MTCNN, extract_face
from icecream import ic
from PIL import Image
from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIPModel,
    CLIPPreTrainedModel,
    CLIPTextTransformer,
    _expand_mask,
)

warnings.filterwarnings("ignore", category=FutureWarning)
inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FastComposerCLIPImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
    ):
        model = CLIPModel.from_pretrained(global_model_name_or_path)
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return FastComposerCLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size

    def forward(self, object_pixel_values):
        b, num_objects, c, h, w = object_pixel_values.shape

        object_pixel_values = object_pixel_values.view(b * num_objects, c, h, w)

        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(object_pixel_values, (h, w), mode="bilinear", antialias=True)

        object_pixel_values = self.vision_processor(object_pixel_values)
        object_embeds = self.vision_model(object_pixel_values)[1]
        object_embeds = self.visual_projection(object_embeds)
        object_embeds = object_embeds.view(b, num_objects, 1, -1)
        return object_embeds


def fuse_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
):
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    seq_length = inputs_embeds.shape[1]
    flat_object_embeds = object_embeds.view(-1, object_embeds.shape[-2], object_embeds.shape[-1])

    valid_object_mask = torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :] < num_objects[:, None]

    valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)

    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds)
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


class FastComposerPostfuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, text_embeds, object_embeds):
        text_object_embeds = torch.cat([text_embeds, object_embeds], dim=-1)
        text_object_embeds = self.mlp1(text_object_embeds) + text_embeds
        text_object_embeds = self.mlp2(text_object_embeds)
        text_object_embeds = self.layer_norm(text_object_embeds)
        return text_object_embeds

    def forward(
        self,
        text_embeds,
        object_embeds,
        image_token_mask,
        num_objects,
    ) -> torch.Tensor:
        text_object_embeds = fuse_object_embeddings(
            text_embeds, image_token_mask, object_embeds, num_objects, self.fuse_fn
        )

        return text_object_embeds


class FastComposerTextEncoder(CLIPPreTrainedModel):
    _build_causal_attention_mask = CLIPTextTransformer._build_causal_attention_mask

    @staticmethod
    def from_pretrained(model_name_or_path, **kwcfg):
        model = CLIPTextModel.from_pretrained(model_name_or_path, **kwcfg)
        text_model = model.text_model
        return FastComposerTextEncoder(text_model)

    def __init__(self, text_model):
        super().__init__(text_model.config)
        self.config = text_model.config
        self.final_layer_norm = text_model.final_layer_norm
        self.embeddings = text_model.embeddings
        self.encoder = text_model.encoder

    def forward(
        self,
        input_ids,
        image_token_mask=None,
        object_embeds=None,
        num_objects=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids)

        bsz, seq_len = input_shape
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import (
        Attention,
        AttnProcessor,
        AttnProcessor2_0,
    )

    UNET_LAYER_NAMES = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(query, key, attention_mask)
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(make_new_get_attention_scores_fn(name), module)

    return unet


class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=1.0, normalize=False):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(dim=2) / background_segmaps_sum

        object_loss = (object_token_attn_prob * object_segmaps).sum(dim=2) / object_segmaps_sum

        return background_loss - object_loss


def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    object_token_idx,
    object_token_idx_mask,
    loss_fn,
):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    # Resize the object segmentation maps to the size of the cross attention scores
    object_segmaps = F.interpolate(
        object_segmaps, size=(size, size), mode="bilinear", antialias=True
    )  # (b, max_num_objects, size, size)

    object_segmaps = object_segmaps.view(b, max_num_objects, -1)  # (b, max_num_objects, num_noise_latents)

    num_heads = bxh // b

    cross_attention_scores = cross_attention_scores.view(b, num_heads, num_noise_latents, num_text_tokens)

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(b, 1, 1, max_num_objects).expand(b, num_heads, num_noise_latents, max_num_objects),
    )  # (b, num_heads, num_noise_latents, max_num_objects)

    object_segmaps = (
        object_segmaps.permute(0, 2, 1).unsqueeze(1).expand(b, num_heads, num_noise_latents, max_num_objects)
    )

    loss = loss_fn(object_token_attn_prob, object_segmaps)

    loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
    object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / object_token_cnt).mean()

    return loss


def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    image_token_idx,
    image_token_idx_mask,
    loss_fn,
):
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn
        )
        loss += layer_loss
    return loss / num_layers


class FastComposerModel(nn.Module):
    def __init__(self, text_encoder: FastComposerTextEncoder, image_encoder, vae, unet, cfg):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.use_ema = False
        self.ema_param = None
        self.pretrained_model_name_or_path = cfg.pretrained_model_path
        self.revision = cfg.revision
        self.non_ema_revision = cfg.non_ema_revision
        self.object_localization: bool = cfg.object_localization
        self.object_localization_weight: float = cfg.object_localization_weight
        self.localization_layers = cfg.localization_layers
        self.mask_loss = cfg.mask_loss
        self.mask_loss_prob = cfg.mask_loss_prob

        self.adapter = LossAdapter()
        self.cfg = cfg
        self.output_dir = cfg.output_dir

        embed_dim = text_encoder.config.hidden_size

        self.postfuse_module = FastComposerPostfuseModule(embed_dim)

        if self.object_localization:
            self.cross_attention_scores = {}
            self.unet = unet_store_cross_attention_scores(
                self.unet, self.cross_attention_scores, self.localization_layers
            )
            self.object_localization_loss_fn = BalancedL1Loss(
                cfg.object_localization_threshold,
                cfg.object_localization_normalize,
            )

    def _clear_cross_attention_scores(self):
        if hasattr(self, "cross_attention_scores"):
            keys = list(self.cross_attention_scores.keys())
            for k in keys:
                del self.cross_attention_scores[k]

        gc.collect()

    @staticmethod
    def from_pretrained(cfg):
        text_encoder = FastComposerTextEncoder.from_pretrained(
            cfg.pretrained_model_path,
            subfolder="text_encoder",
            revision=cfg.revision,
        )
        vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae", revision=cfg.revision)
        unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_model_path,
            subfolder="unet",
            revision=cfg.non_ema_revision,
        )
        image_encoder = FastComposerCLIPImageEncoder.from_pretrained(
            cfg.image_encoder_name_or_path,
        )

        return FastComposerModel(text_encoder, image_encoder, vae, unet, cfg)

    def to_pipeline(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            revision=self.revision,
            non_ema_revision=self.non_ema_revision,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
        )
        pipe.safety_checker = None

        pipe.image_encoder = self.image_encoder

        pipe.postfuse_module = self.postfuse_module

        return pipe

    def forward(self, batch, noise_scheduler, return_image=False):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        image_token_mask = batch["image_token_mask"]
        object_pixel_values = batch["object_pixel_values"]
        num_objects = batch["num_objects"]

        vae_dtype = self.vae.parameters().__next__().dtype
        vae_input = pixel_values.to(vae_dtype)

        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # (bsz, max_num_objects, num_image_tokens, dim)
        object_embeds = self.image_encoder(object_pixel_values)

        encoder_hidden_states = self.text_encoder(input_ids, image_token_mask, object_embeds, num_objects)[
            0
        ]  # (bsz, seq_len, dim)

        encoder_hidden_states = self.postfuse_module(
            encoder_hidden_states,
            object_embeds,
            image_token_mask,
            num_objects,
        )

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.mask_loss and torch.rand(1) < self.mask_loss_prob:
            object_segmaps = batch["object_segmaps"]
            mask = (object_segmaps.sum(dim=1) > 0).float()
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(pred.shape[-2], pred.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred = pred * mask
            target = target * mask

        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        return_dict = {"denoise_loss": denoise_loss}

        if self.object_localization:
            object_segmaps = batch["object_segmaps"]
            image_token_idx = batch["image_token_idx"]
            image_token_idx_mask = batch["image_token_idx_mask"]
            localization_loss = get_object_localization_loss(
                self.cross_attention_scores,
                object_segmaps,
                image_token_idx,
                image_token_idx_mask,
                self.object_localization_loss_fn,
            )
            return_dict["localization_loss"] = localization_loss
            loss = self.object_localization_weight * localization_loss + denoise_loss
            self._clear_cross_attention_scores()
        else:
            loss = denoise_loss

        decoded_gen_image = self.generate_images(latents, timesteps, encoder_hidden_states, noise_scheduler, 25)

        if return_image:
            to_pil = T.ToPILImage()
            ref_img_pil = to_pil(pixel_values[0].cpu().float().clamp(0, 1))
            gen_img_pil = to_pil(decoded_gen_image[0].cpu().float().clamp(0, 1))
            return_dict["decoded_ref_image"] = ref_img_pil
            return_dict["decoded_gen_image"] = gen_img_pil

        pred = pred.to(target.device)

        if self.cfg.face_separation:  # face separation loss
            face_errors = []
            for i, (ref_image_tensor, gen_image_tensor) in enumerate(zip(pixel_values, decoded_gen_image)):
                ref_image_pil = self.adapter.tensor_to_pil(ref_image_tensor)
                gen_image_pil = self.adapter.tensor_to_pil(gen_image_tensor)

                ref_boxes, _ = self.adapter.detect(ref_image_pil, landmarks=False)
                gen_boxes, _ = self.adapter.detect(gen_image_pil, landmarks=False)

                if ref_boxes is None or gen_boxes is None:
                    face_errors.append(torch.tensor(0.0))
                    continue

                ref_faces = []
                gen_faces = []
                for j, box in enumerate(ref_boxes):
                    tensor_face = extract_face(ref_image_pil, box)
                    ref_faces.append(tensor_face)
                for j, box in enumerate(gen_boxes):
                    tensor_face = extract_face(gen_image_pil, box)
                    gen_faces.append(tensor_face)

                for ref_face, gen_face in zip(ref_faces, gen_faces):
                    ref_face = ref_face.to(gen_face.device).float()
                    gen_face = gen_face.float()

                    mse_loss = F.mse_loss(gen_face, ref_face, reduction="mean")
                    face_errors.append(mse_loss)

            face_separation_loss = torch.stack(face_errors).mean() * self.cfg.face_separation_weight
            return_dict["face_separation_loss"] = face_separation_loss
            loss += face_separation_loss
            print(f"face_separation_loss: {torch.stack(face_errors).mean()} >> scaled: {face_separation_loss}")

        if self.cfg.expression_separation:  # expression separation loss
            continue
        
        if self.cfg.landmark_separation:  # landmark separation loss
            landmark_errors = []
            for i, (ref_image_tensor, gen_image_tensor) in enumerate(zip(pixel_values, decoded_gen_image)):
                ref_image_pil = self.adapter.tensor_to_pil(ref_image_tensor)
                gen_image_pil = self.adapter.tensor_to_pil(gen_image_tensor)

                ref_boxes, _, ref_points = self.adapter.detect(ref_image_pil, landmarks=True)
                gen_boxes, _, gen_points = self.adapter.detect(gen_image_pil, landmarks=True)

                if ref_boxes is None or gen_boxes is None:
                    landmark_errors.append(torch.tensor(0.0))
                    continue

                # calculate the distance between the points
                for ref_point, gen_point in zip(ref_points, gen_points):
                    ref_point = torch.tensor(ref_point).to(gen_point.device).float()
                    gen_point = gen_point.float()

                    mse_loss = F.mse_loss(gen_point, ref_point, reduction="mean")
                    landmark_errors.append(mse_loss)

            landmark_separation_loss = torch.stack(landmark_errors).mean() * self.cfg.landmark_separation_weight
            return_dict["landmark_separation_loss"] = landmark_separation_loss
            loss += landmark_separation_loss
            print(f"landmark_separation_loss: {torch.stack(landmark_errors).mean()} >> scaled: {landmark_separation_loss}")

        if self.cfg.identity_separation:  # identity separation loss
            identity_errors = []
            for i, (ref_image_tensor, gen_image_tensor) in enumerate(zip(pixel_values, decoded_gen_image)):
                ref_image_pil = self.adapter.tensor_to_pil(ref_image_tensor)
                gen_image_pil = self.adapter.tensor_to_pil(gen_image_tensor)

                identity_error = self.adapter.identity_loss(ref_image_pil, gen_image_pil)
                identity_errors.append(identity_error)
        
            identity_separation_loss = torch.stack(identity_errors).mean()
            return_dict["identity_separation_loss"] = identity_separation_loss
            loss += identity_separation_loss
            print(f"identity_separation_loss: {torch.stack(identity_errors).mean()} >> scaled: {identity_separation_loss}")


        return_dict["loss"] = loss
        print(f"loss: {loss}, denoise_loss: {denoise_loss}, localization_loss: {localization_loss}")
        torch.cuda.empty_cache()
        return return_dict

    @torch.no_grad()
    def generate_images(self, latents, timesteps, encoder_hidden_states, scheduler, num_inference_steps):
        for i, t in enumerate(timesteps):
            latent_model_input = scheduler.scale_model_input(latents, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample

        return image


class LossAdapter:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(margin=0, keep_all=True, device=self.device)
        self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.to_pil = T.ToPILImage()

    def _detect_faces(self, image_pil: Image) -> torch.Tensor:
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        return self.mtcnn(image_pil)

    def detect(self, image, landmarks=False):
        return self.model.detect(image, landmarks=landmarks)

    def _get_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        return self.embedding_model(image_tensor.unsqueeze(0).to(self.device)).detach().cpu().numpy().flatten()

    def _calc_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def calc_face_similarity(self, image1: Image, image2: Image) -> float:
        image1_tensor = self._detect_faces(image1)[0]
        image2_tensor = self._detect_faces(image2)[0]
        embedding1 = self._get_embedding(image1_tensor)
        embedding2 = self._get_embedding(image2_tensor)

        return self._calc_similarity(embedding1, embedding2)

    def tensor_to_pil(self, tensor_image):
        return self.to_pil(tensor_image.cpu().float().clamp(0, 1))

    def calc_face_essential_similarity(self, image1: Image, image2: Image) -> float:
        # rotate image
        stack_embedding_image1 = []
        stack_embedding_image2 = []
        for angle in range(0, 360, 10):
            times = [time.time()]
            image1_rotated = image1.rotate(angle)
            image2_rotated = image2.rotate(angle)
            times.append(time.time())
            # print(f"rotate time: {times[-1] - times[-2]}")
            image1_tensor = self._detect_faces(image1_rotated)
            image2_tensor = self._detect_faces(image2_rotated)
            times.append(time.time())
            # print(f"detect time: {times[-1] - times[-2]}")
            if image1_tensor is None or image2_tensor is None:
                continue
            embedding1 = self._get_embedding(image1_tensor[0])
            embedding2 = self._get_embedding(image2_tensor[0])
            times.append(time.time())
            # print(f"embedding time: {times[-1] - times[-2]}")
            stack_embedding_image1.append(embedding1)
            stack_embedding_image2.append(embedding2)

        if not stack_embedding_image1 or not stack_embedding_image2:
            raise ValueError("No embeddings could be calculated from the rotated images.")

        # average
        mean_embedding_image1 = np.mean(stack_embedding_image1, axis=0)
        mean_embedding_image2 = np.mean(stack_embedding_image2, axis=0)

        return self._calc_similarity(mean_embedding_image1, mean_embedding_image2)

    def identity_loss(self, image1: Image, image2: Image) -> float:
        return abs(calc_face_essential_similarity(image1, image2) - self.calc_face_similarity(image1, image2))


def extract_focus_region(image, point, face_rate):
    """
    画像からポイントの周囲を face_rate に基づいて矩形領域として切り取る関数。
    """
    x, y = int(point[0]), int(point[1])
    box_size = int(min(image.width, image.height) * face_rate)

    left = min(max(x - box_size // 2, 0), image.width)
    top = min(max(y - box_size // 2, 0), image.height)
    right = max(min(x + box_size // 2, image.width), 0)
    bottom = max(min(y + box_size // 2, image.height), 0)

    try:
        focus_region = image.crop((left, top, right, bottom))
    except Exception as e:
        ic(image.width, image.height)
        ic(x, y, box_size)
        ic(left, top, right, bottom)
        raise e

    return focus_region
