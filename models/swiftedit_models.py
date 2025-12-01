import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
)
from typing import Optional, Tuple, List
from PIL import Image

from .image_projection import ImageProjectionModel
from .attention_processors import StandardAttnProcessor, IPAdapterAttnProcessor
from .mask_attention_processor import MaskAwareIPAttnProcessor
from .mask_controller import MaskController
from utils.editing_utils import preprocess_image_for_vae


def tokenize_captions(tokenizer, captions: List[str]) -> torch.Tensor:
    """
    Tokenize text captions for CLIP text encoder.
    """
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


class AuxiliaryModels:
    """
    VAE for image encoding/decoding
    CLIP text encoder
    CLIP image encoder
    Noise scheduler
    Tokenizer
    """

    # hf model paths taken from ref impl
    def __init__(
        self,
        base_model_name: str = "sd2-community/stable-diffusion-2-1-base",
        image_encoder_path: str = "h94/IP-Adapter",
        device: str = "cuda",
    ):
        self.device = device

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            base_model_name, subfolder="scheduler"
        )

        self.vae_decoder = AutoencoderKL.from_pretrained(
            base_model_name, subfolder="vae"
        ).to(device, dtype=torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model_name, subfolder="text_encoder"
        ).to(device, dtype=torch.float32)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, subfolder="models/image_encoder"
        ).to(device, dtype=torch.float32)
        self.image_encoder.requires_grad_(False)

        self.clip_image_processor = CLIPImageProcessor()

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        input_ids = tokenize_captions(self.tokenizer, prompts).to(self.device)
        text_embeddings = self.text_encoder(input_ids)[0]
        return text_embeddings

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        clip_images = self.clip_image_processor(
            images=images, return_tensors="pt").pixel_values
        clip_images = clip_images.to(self.device, dtype=torch.float32)
        image_embeddings = self.image_encoder(clip_images).image_embeds
        return image_embeddings


class InversionModel:
    """
    Impl of one-step inversion network F, which transforms latent z into inverted noise eps_hat

    Args:
        pretrained_model_path: Path to pretrained inversion network
        base_model_name: Base model for noise scheduler and VAE
        dtype: Weight dtype
        device: Device to load on
    """

    # base models paths taken from reference impl
    def __init__(
        self,
        pretrained_model_path: str,
        base_model_name: str = "stabilityai/sd-turbo",
        dtype: str = "fp16",
        device: str = "cuda",
    ):
        if dtype == "fp16":
            self.weight_dtype = torch.float16
        elif dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

        self.device = device
        self.base_model_name = base_model_name

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            base_model_name, subfolder="scheduler"
        )

        self.vae_encoder = AutoencoderKL.from_pretrained(
            base_model_name, subfolder="vae"
        ).to(device, dtype=torch.float32)

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_path, subfolder="unet_ema"
        ).to(device, dtype=self.weight_dtype)
        self.unet.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model_name, subfolder="text_encoder"
        ).to(device, dtype=self.weight_dtype)

        # compute corruption noise parameters
        timestep = torch.ones((1,), dtype=torch.int64, device=device)
        timestep = timestep * \
            (self.noise_scheduler.config.num_train_timesteps - 1)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)

        corruption_timestep = int(timestep / 4)
        self.corrupt_alpha_t = (
            alphas_cumprod[corruption_timestep] ** 0.5).view(-1, 1, 1, 1)
        self.corrupt_sigma_t = (
            (1 - alphas_cumprod[corruption_timestep]) ** 0.5).view(-1, 1, 1, 1)

        del alphas_cumprod

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode PIL image to latent

        Returns:
            Latent representation, shape (1, 4, H/8, W/8)
        """
        pixel_values = preprocess_image_for_vae(image, device=self.device)
        latent = self.vae_encoder.encode(pixel_values).latent_dist.sample() * self.vae_encoder.config.scaling_factor
        return latent

    @torch.no_grad()
    def invert(
        self,
        image_latent: torch.Tensor,
        prompt: str,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Invert image latent to noise.

        Returns:
            Inverted noise eps_hat, shape (batch_size, 4, H/8, W/8)
        """
        input_ids = tokenize_captions(self.tokenizer, [prompt]).to(self.device)
        prompt_embeds = self.text_encoder(input_ids)[0]

        if timestep is None:
            timestep = self.noise_scheduler.config.num_train_timesteps - 1

        inverted_noise = self.unet(
            image_latent,
            torch.tensor([timestep], device=self.device),
            prompt_embeds,
        ).sample

        return inverted_noise


class AttentionCapturingWrapper:
    def __init__(self, original_processor, layer_name):
        self.original_processor = original_processor
        self.layer_name = layer_name
        self.attention_map = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        # Only capture attention for cross-attention layers (encoder_hidden_states must be provided)
        if encoder_hidden_states is None:
            # This is self-attention, skip capturing
            return self.original_processor(attn, hidden_states, encoder_hidden_states, **kwargs)

        # Handle 4D inputs (same as original processor)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states_flat = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, _, _ = hidden_states.shape
            hidden_states_flat = hidden_states

        # Extract text embeddings (exclude image tokens if present)
        # For IP-Adapter, last 4 tokens are image features
        if hasattr(self.original_processor, 'num_tokens'):
            num_tokens = self.original_processor.num_tokens
            if encoder_hidden_states.shape[1] > num_tokens:
                end_pos = encoder_hidden_states.shape[1] - num_tokens
                text_hidden_states = encoder_hidden_states[:, :end_pos, :]
            else:
                text_hidden_states = encoder_hidden_states
        else:
            text_hidden_states = encoder_hidden_states

        # Apply normalization if needed (before computing Q, K)
        if attn.norm_cross:
            text_hidden_states = attn.norm_encoder_hidden_states(text_hidden_states)

        # Debug: verify we have text_hidden_states
        if text_hidden_states is None:
            print(f"Warning: text_hidden_states is None for {self.layer_name}")
            return self.original_processor(attn, hidden_states, encoder_hidden_states, **kwargs)

        try:
            # Get query and key for text cross-attention
            query = attn.to_q(hidden_states_flat)
            key = attn.to_k(text_hidden_states)

            # Reshape for multi-head attention
            head_dim = query.shape[-1] // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # (B, H, N, D)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # (B, H, M, D)

            # Compute attention similarity: Q @ K^T
            # Shape: (B, H, N, M) where N=spatial_tokens, M=text_tokens
            attention_sim = torch.einsum('b h i d, b h j d -> b h i j', query, key) * attn.scale

            # Store attention map (before softmax)
            self.attention_map = attention_sim.detach()
        except Exception as e:
            print(f"Error in wrapper {self.layer_name}: {e}")
            import traceback
            traceback.print_exc()

        # Continue with original processor
        return self.original_processor(attn, hidden_states, encoder_hidden_states, **kwargs)


class IPSBv2Model(nn.Module):
    """
    SwiftBrushv2 model with IP-Adapter (G^IP)

    Args:
        pretrained_model_path: Path to SBv2 UNet
        ip_adapter_path: Path to trained IP-Adapter weights
        aux_models: Auxiliary models instance
        device: Device
        use_mask_controller: Whether to use mask-aware attention processors
    """

    def __init__(
        self,
        pretrained_model_path: str,
        ip_adapter_path: str,
        aux_models: AuxiliaryModels,
        device: str = "cuda",
        use_mask_controller: bool = False,
    ):
        super().__init__()

        self.device = device
        self.aux_models = aux_models
        self.use_mask_controller = use_mask_controller

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_path
        ).to(device)
        self.unet.eval()

        self.timestep = torch.ones((1,), dtype=torch.int64, device=device)
        self.timestep = self.timestep * (
            aux_models.noise_scheduler.config.num_train_timesteps - 1
        )

        self.image_proj_model = ImageProjectionModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=aux_models.image_encoder.config.projection_dim,
            num_tokens=4,  # N=4 from paper
        ).to(device)

        self._initialize_attention_processors()

        self.load_state_dict(torch.load(ip_adapter_path, map_location=device))

        # compute alpha_t and sigma_t for one-step denoising
        alphas_cumprod = aux_models.noise_scheduler.alphas_cumprod.to(device)
        self.alpha_t = (alphas_cumprod[self.timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = (
            (1 - alphas_cumprod[self.timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod

    def _initialize_attention_processors(self):
        """
        Initialize (decoupled) attention processors with IP-Adapter.
        """
        attn_procs = {}
        unet_state_dict = self.unet.state_dict()

        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if cross_attention_dim is None:
                # self attn
                attn_procs[name] = StandardAttnProcessor()
            else:
                # cross attn
                layer_name = name.split(".processor")[0]

                weights = {
                    "to_k_ip.weight": unet_state_dict[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_state_dict[layer_name + ".to_v.weight"],
                }

                if self.use_mask_controller:
                    processor = MaskAwareIPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )
                else:
                    processor = IPAdapterAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )

                processor.load_state_dict(weights)
                attn_procs[name] = processor.to(self.device)

        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = nn.ModuleList(
            self.unet.attn_processors.values())

    @torch.no_grad()
    def get_image_embeds(
        self,
        pil_images: Optional[List[Image.Image]] = None,
        clip_image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get image prompt embeddings
        """
        if pil_images is not None:
            clip_image_embeds = self.aux_models.encode_image(pil_images)
        elif clip_image_embeds is not None:
            clip_image_embeds = clip_image_embeds.to(
                self.device, dtype=torch.float32)
        else:
            raise ValueError(
                "Either pil_images or clip_image_embeds must be provided")

        # Project to image prompt embeddings
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

    def set_scale(self, scale: float):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, MaskAwareIPAttnProcessor)):
                attn_processor.scale = scale

    def set_mask_controller(
        self,
        controller: Optional[MaskController],
        where: List[str] = ["down_blocks", "mid_block", "up_blocks"],
    ):
        """
        Set mask controller for ARaM editing.
        """
        if not self.use_mask_controller:
            return

        for name, attn_processor in self.unet.attn_processors.items():
            if isinstance(attn_processor, MaskAwareIPAttnProcessor):
                should_set = any(block_name in name for block_name in where)
                if should_set:
                    attn_processor.controller = controller

    @torch.no_grad()
    def collect_attention_maps(
        self,
        noisy_latent: torch.Tensor,
        prompt: str,
        source_image: Optional[Image.Image] = None,
        scale: float = 1.0,
    ) -> List[torch.Tensor]:
        """
        Collect cross-attention maps from UNet forward pass.

        Args:
            noisy_latent: Noisy latent tensor, shape (1, 4, H/8, W/8)
            prompt: Text prompt for attention computation
            source_image: Optional source image for IP-Adapter
            scale: Image condition scale

        Returns:
            List of attention maps, one per cross-attention layer (mid_block, up_blocks)
            Each map shape: (batch=1, heads, spatial_tokens, text_tokens)
        """
        self.set_scale(scale)

        # Prepare prompt embeddings
        text_embeds = self.aux_models.encode_text([prompt])

        if source_image is not None:
            image_prompt_embeds = self.get_image_embeds(pil_images=[source_image])
            prompt_embeds = torch.cat([text_embeds, image_prompt_embeds], dim=1)
        else:
            prompt_embeds = text_embeds

        # Storage for attention maps
        attention_maps = []

        # Replace processors with wrappers for mid_block and up_blocks only
        original_processors = {}
        wrapped_processors = {}
        wrapped_count = 0


        for name, processor in self.unet.attn_processors.items():
            # Store original processor
            original_processors[name] = processor

            # Only wrap cross-attention layers in mid_block and up_blocks
            # (skip self-attention and down_blocks)
            # Cross-attention layers have "attn2" in the name, self-attention has "attn1"
            if (name.startswith("mid_block") or name.startswith("up_blocks")) and "attn2" in name:
                wrapped_processors[name] = AttentionCapturingWrapper(processor, name)
                wrapped_count += 1
            else:
                # Keep original processor for self-attention and down_blocks
                wrapped_processors[name] = processor


        # Set wrapped processors
        self.unet.set_attn_processor(wrapped_processors)

        # Run forward pass
        _ = self.unet(noisy_latent, self.timestep, prompt_embeds).sample

        # Collect attention maps from the UNet's processors
        for name, processor in self.unet.attn_processors.items():
            if isinstance(processor, AttentionCapturingWrapper) and processor.attention_map is not None:
                attention_maps.append(processor.attention_map)

        self.unet.set_attn_processor(original_processors)
        return attention_maps

    @torch.no_grad()
    def generate(
        self,
        inverted_noise: torch.Tensor,
        prompts: List[str],
        source_image: Optional[Image.Image] = None,
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate image from inverted noise.

        From paper: z^edit = G^IP(eps_hat, c^edit_y, c_x)

        Args:
            inverted_noise: Inverted noise eps_hat, shape (1, 4, H/8, W/8)
            prompts: List of target prompts
            source_image: Source image for IP-Adapter conditioning
            scale: Image condition scale

        Returns:
            Tuple of (generated_image, noise_visualization)
        """
        self.set_scale(scale)
        num_samples = len(prompts)

        # get image prompt embeddings
        if source_image is not None:
            image_prompt_embeds = self.get_image_embeds(
                pil_images=[source_image])
            batch_size, seq_len, dim = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(
                batch_size * num_samples, seq_len, dim)
        else:
            image_prompt_embeds = None

        # text embeddings
        text_embeds = self.aux_models.encode_text(prompts)

        if image_prompt_embeds is not None:
            prompt_embeds = torch.cat(
                [text_embeds, image_prompt_embeds], dim=1)
        else:
            prompt_embeds = text_embeds

        noise = inverted_noise.repeat(num_samples, 1, 1, 1)

        # one step denoise
        model_pred = self.unet(noise, self.timestep, prompt_embeds).sample

        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

        # predicted original sample
        pred_original_sample = (noise - self.sigma_t *
                                model_pred) / self.alpha_t

        # thresholding & clip
        if self.aux_models.noise_scheduler.config.thresholding:
            pred_original_sample = self.aux_models.noise_scheduler._threshold_sample(
                pred_original_sample
            )
        elif self.aux_models.noise_scheduler.config.clip_sample:
            clip_range = self.aux_models.noise_scheduler.config.clip_sample_range
            pred_original_sample = pred_original_sample.clamp(
                -clip_range, clip_range)

        pred_original_sample = pred_original_sample / \
            self.aux_models.vae_decoder.config.scaling_factor
        images = self.aux_models.vae_decoder.decode(
            pred_original_sample.to(dtype=torch.float32)
        ).sample

        images = (images + 1.0) / 2.0

        noise_vis = noise / self.aux_models.vae_decoder.config.scaling_factor
        noise_images = self.aux_models.vae_decoder.decode(
            noise_vis.to(dtype=torch.float32)
        ).sample
        noise_images = (noise_images + 1.0) / 2.0

        return images, noise_images
