from utils.editing_utils import (
    extract_editing_mask,
    extract_attention_based_mask,
    postprocess_image,
    visualize_mask,
    create_side_by_side,
)
from models.mask_controller import MaskController
from models.swiftedit_models import (
    InversionModel,
    IPSBv2Model,
    AuxiliaryModels,
)
import torch
import sys
import os
import argparse
from pathlib import Path
from PIL import Image
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SwiftEditPipeline:
    def __init__(
        self,
        inversion_model_path: str,
        sbv2_model_path: str,
        ip_adapter_path: str,
        base_model_name: str = "sd2-community/stable-diffusion-2-1-base",
        inversion_base_model: str = "stabilityai/sd-turbo",
        device: str = "cuda",
        dtype: str = "fp16",
    ):
        """
        Args:
            inversion_model_path: Path to trained inversion network (e.g., inverse_ckpt-120k)
            sbv2_model_path: Path to SwiftBrushV2 base model (e.g., sbv2_0.5)
            ip_adapter_path: Path to trained IP-Adapter weights (e.g., ip_adapter.bin)
            base_model_name: Base diffusion model for auxiliary models (SD 2.1)
            inversion_base_model: Base model for inversion network (SD-Turbo)
            device: Device to run on
            dtype: Weight dtype
        """
        print("Initializing SwiftEdit pipeline...")

        self.device = device
        print("Loading auxiliary model...")
        self.aux_models = AuxiliaryModels(
            base_model_name=base_model_name,
            device=device,
        )

        print("Loading inversion model...")
        self.inversion_model = InversionModel(
            pretrained_model_path=inversion_model_path,
            base_model_name=inversion_base_model,
            dtype=dtype,
            device=device,
        )

        print("Loading IP-SBv2 model...")
        self.ip_sbv2_model = IPSBv2Model(
            pretrained_model_path=sbv2_model_path,
            ip_adapter_path=ip_adapter_path,
            aux_models=self.aux_models,
            device=device,
            use_mask_controller=True,
        )

        print("SwiftEdit ready")

    @torch.no_grad()
    def edit(
        self,
        source_image: str | Image.Image,
        source_prompt: str,
        edit_prompt: str,
        edit_mask: Optional[torch.Tensor] = None,
        use_self_guided_mask: bool = True,
        mask_method: str = "noise",
        scale_edit: float = 0.2,
        scale_non_edit: float = 1.0,
        scale_text: float = 1.0,
        return_intermediate: bool = False,
    ) -> Image.Image | dict:
        """
        This function edits the image in accordance w/ the paper as follows:

        As in the paper, x is the input image, z represents the latent, c_source is the source prompt, c_edit is the edit prompt

        1. Encode source image: z^source = E(x^source)
        2. One-step inversion: eps_hat = F(z^source, c^source_y)
        3. Extract mask
        4. One-step editing: z^edit = G^IP(eps_hat, c^edit_y, c_x) with ARaM
        5. Decode: x^edit = D(z^edit)

        Args (defaults are taken from the reference implementation):
            source_image: Path to image or PIL Image
            source_prompt: Source description
            edit_prompt: Edit description
            edit_mask: Optional pre-defined mask
            use_self_guided_mask: Extract mask automatically
            mask_method: Mask extraction method - "noise" (default) or "attention"
            scale_edit: Image condition scale in edit region
            scale_non_edit: Image condition scale in non-edit region
            scale_text: Text-alignment scale
            return_intermediate: Return intermediate results (return will be a dict)

        Returns:
            Edited PIL Image
        """
        if isinstance(source_image, str):
            source_image = Image.open(source_image).convert("RGB")
        elif not isinstance(source_image, Image.Image):
            raise ValueError("source_image must be path or PIL Image")

        print(f"Editing image...")
        print(f"  Source prompt: {source_prompt}")
        print(f"  Edit prompt: {edit_prompt}")

        # Encode source image
        print("Step 1/5: Encoding source image...")
        image_latent = self.inversion_model.encode_image(source_image)

        # One-step inversion
        print("Step 2/5: Inverting to noise (one-step)...")
        inverted_noise = self.inversion_model.invert(
            image_latent, source_prompt)

        # Construct noisy latent (needed for both mask extraction and generation)
        noisy_latent = (
            self.ip_sbv2_model.alpha_t * image_latent +
            self.ip_sbv2_model.sigma_t * inverted_noise
        )

        # Extract editing mask
        print("Step 3/5: Extracting editing mask...")
        if edit_mask is None and use_self_guided_mask:
            if mask_method == "attention":
                edit_mask = extract_attention_based_mask(
                    ip_sbv2_model=self.ip_sbv2_model,
                    noisy_latent=noisy_latent,
                    source_prompt=source_prompt,
                    edit_prompt=edit_prompt,
                    source_image=source_image,
                    threshold=0.5,
                    clamp_rate=3.0,
                )
            else:  # mask_method == "noise" (default)
                edit_mask = extract_editing_mask(
                    inversion_model=self.inversion_model,
                    image_latent=image_latent,
                    source_prompt=source_prompt,
                    edit_prompt=edit_prompt,
                    threshold=0.5,
                    clamp_rate=3.0,
                    mid_timestep=500,
                )
            print(
                f"  Mask extracted (method: {mask_method}, coverage: {(edit_mask > 0.5).float().mean().item():.2%})")
        elif edit_mask is None:
            edit_mask = torch.ones(
                1, 1, image_latent.shape[2], image_latent.shape[3],
                device=self.device
            )

        # Construct noisy latent (already done above, but keeping for clarity)
        print("Step 4/5: Constructing noisy latent...")

        # Setup mask controller for ARaM
        print("Step 5/5: Setting up ARaM and generating...")
        mask_controller = MaskController(
            mask=edit_mask,
            scale_text=scale_text,
            scale_edit=scale_edit,
            scale_non_edit=scale_non_edit,
        )
        self.ip_sbv2_model.set_mask_controller(
            mask_controller,
            where=["mid_blocks", "up_blocks"],
        )

        edited_images, _ = self.ip_sbv2_model.generate(
            inverted_noise=noisy_latent,
            prompts=[source_prompt, edit_prompt],
            source_image=source_image,
            scale=1.0,
        )

        edited_image = postprocess_image(edited_images[1:2])

        print("Editing complete")

        if return_intermediate:
            return {
                "edited_image": edited_image,
                "source_image": source_image,
                "edit_mask": edit_mask,
                "inverted_noise": inverted_noise,
                "image_latent": image_latent,
            }
        else:
            return edited_image

    def create_visualization(
        self,
        source_image: Image.Image,
        edited_image: Image.Image,
        edit_mask: torch.Tensor,
    ) -> Image.Image:
        """
        Create side-by-side visualization of editing results.

        Args:
            source_image: Original image
            edited_image: Edited image
            edit_mask: Editing mask

        Returns:
            Combined visualization image
        """
        mask_vis = visualize_mask(edit_mask)
        comparison = create_side_by_side(
            images=[source_image, mask_vis, edited_image],
            labels=["Source", "Edit Mask", "Edited"],
        )

        return comparison
