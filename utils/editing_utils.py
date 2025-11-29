import torch
import numpy as np
from PIL import Image
from typing import Optional
import torchvision.transforms as transforms


def extract_editing_mask(
    inversion_model,
    image_latent: torch.Tensor,
    source_prompt: str,
    edit_prompt: str,
    threshold: float = 0.5,
    clamp_rate: float = 3.0,
    mid_timestep: int = 500,
) -> torch.Tensor:
    """
    Extract self-guided editing mask.

    Algo taken from paper:
    1. eps_source = F(z^source, c^source_y, t=500)
    2. eps_edit = F(z^source, c^edit_y, t=500)
    3. diff = |eps_source - eps_edit|.mean(channels)
    4. M = clamp(diff, 0, diff.mean() * clamp_rate) / (diff.mean() * clamp_rate)
    5. M = binarize(M, threshold)

    Args:
        inversion_model: Trained inversion network
        image_latent: Source image latent z^source
        source_prompt: Source description
        edit_prompt: Edit description
        threshold: Threshold for binary mask 
        clamp_rate: Clamping multiplier
        mid_timestep: Timestep for mask extraction

    Returns:
        Editing mask
    """
    # invert with source prompt at mid timestep
    noise_source = inversion_model.invert(
        image_latent, source_prompt, timestep=mid_timestep)

    # invert with edit prompt at mid timestep
    noise_edit = inversion_model.invert(
        image_latent, edit_prompt, timestep=mid_timestep)

    # compute difference
    noise_diff = torch.abs(noise_source - noise_edit)

    # average across batch and channels dimensions
    mask = noise_diff.mean(dim=[0, 1], keepdim=False)

    # normalize
    max_v = (mask.mean() * clamp_rate).item()
    if max_v > 0:
        mask = mask.clamp(0, max_v) / max_v
    else:
        mask = torch.zeros_like(mask)

    def to_binary(pix, thresh=threshold):
        return 1.0 if float(pix) > thresh else 0.0

    # apply thresholding
    mask = mask.detach().cpu().apply_(lambda pix: to_binary(pix, threshold))
    mask = mask.to(inversion_model.device)

    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


def postprocess_image(
    tensor: torch.Tensor,
) -> Image.Image:
    """
    tensor -> image
    """

    tensor = tensor.squeeze(0).clamp(0, 1)
    tensor = tensor.cpu()
    image = transforms.ToPILImage()(tensor)

    return image


def visualize_mask(
    mask: torch.Tensor,
    colormap: str = "viridis",
) -> Image.Image:
    """
    visualize editing mask
    """
    from matplotlib import cm

    mask_np = mask.squeeze().cpu().numpy()
    cmap = cm.get_cmap(colormap)
    mask_colored = cmap(mask_np)
    mask_colored = (mask_colored[:, :, :3] * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_colored)

    return mask_image


def create_side_by_side(
    images: list,
    labels: Optional[list] = None,
) -> Image.Image:
    """
    create side-by-side comparison of images.
    """
    if not images:
        raise ValueError("No images provided")

    widths, heights = zip(*[img.size for img in images])
    total_width = sum(widths)
    max_height = max(heights)

    combined = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    if labels:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(combined)
        font = ImageFont.load_default()

        x_offset = 0
        for img, label in zip(images, labels):
            draw.text((x_offset + 10, 10), label, fill='white', font=font)
            x_offset += img.width

    return combined
