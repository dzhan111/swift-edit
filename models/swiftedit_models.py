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

        self.vae = AutoencoderKL.from_pretrained(
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

        self.vae = AutoencoderKL.from_pretrained(
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
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        pixel_values = transform(image).unsqueeze(
            0).to(self.device, dtype=torch.float32)

        latent = self.vae.encode(pixel_values).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor

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
            self.aux_models.vae.config.scaling_factor
        images = self.aux_models.vae.decode(
            pred_original_sample.to(dtype=torch.float32)
        ).sample

        images = (images + 1.0) / 2.0

        noise_vis = noise / self.aux_models.vae.config.scaling_factor
        noise_images = self.aux_models.vae.decode(
            noise_vis.to(dtype=torch.float32)
        ).sample
        noise_images = (noise_images + 1.0) / 2.0

        return images, noise_images
