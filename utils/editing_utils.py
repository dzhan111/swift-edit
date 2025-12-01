import torch
import numpy as np
from PIL import Image
from typing import Optional, List
import torchvision.transforms as transforms
import torch.nn.functional as F

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
    def to_binary(pix, thresh=threshold):
        return 1.0 if float(pix) > thresh else 0.0

    noise_source = inversion_model.invert(
        image_latent, source_prompt, timestep=mid_timestep)
    noise_edit = inversion_model.invert(
        image_latent, edit_prompt, timestep=mid_timestep)

    noise_diff = torch.abs(noise_source - noise_edit)
    mask = noise_diff.mean(dim=[0, 1], keepdim=False)
    max_v = (mask.mean() * clamp_rate).item()
    if max_v > 0:
        mask = mask.clamp(0, max_v) / max_v
    else:
        mask = torch.zeros_like(mask)

    mask = mask.detach().cpu().apply_(lambda pix: to_binary(pix, threshold))
    mask = mask.to(inversion_model.device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def extract_attention_based_mask(
    ip_sbv2_model,
    noisy_latent: torch.Tensor,
    source_prompt: str,
    edit_prompt: str,
    source_image: Optional[Image.Image] = None,
    threshold: float = 0.5,
    clamp_rate: float = 3.0,
) -> torch.Tensor:
    """
    Extract editing mask using cross-attention maps from SBv2 generator.

    Steps:
    1. Run forward pass with source prompt → collect attention maps A_source
    2. Run forward pass with edit prompt → collect attention maps A_edit
    3. Compute attention difference: Δ_attn = |A_source - A_edit|
    4. Aggregate over heads, text tokens, and layers
    5. Normalize and threshold

    Args:
        ip_sbv2_model: IPSBv2Model instance
        noisy_latent: Noisy latent tensor, shape (1, 4, H/8, W/8)
        source_prompt: Source description
        edit_prompt: Edit description
        source_image: Optional source image for IP-Adapter
        threshold: Threshold for binary mask
        clamp_rate: Clamping multiplier for normalization

    Returns:
        Editing mask, shape (1, 1, H, W)
    """
    def to_binary(pix, thresh=threshold):
        return 1.0 if float(pix) > thresh else 0.0

    attn_maps_source = ip_sbv2_model.collect_attention_maps(
        noisy_latent=noisy_latent,
        prompt=source_prompt,
        source_image=source_image,
        scale=1.0,
    )

    attn_maps_edit = ip_sbv2_model.collect_attention_maps(
        noisy_latent=noisy_latent,
        prompt=edit_prompt,
        source_image=source_image,
        scale=1.0,
    )

    if len(attn_maps_source) != len(attn_maps_edit):
        raise ValueError(f"Mismatch in number of attention layers: {len(attn_maps_source)} vs {len(attn_maps_edit)}")

    # Compute attention difference per layer
    attn_diffs = []
    for attn_src, attn_edit in zip(attn_maps_source, attn_maps_edit):
        attn_diff = torch.abs(attn_src - attn_edit)  # (1, H, N, M)
        attn_diff = attn_diff.mean(dim=1)  # (1, N, M)
        attn_diff = attn_diff.mean(dim=-1)  # (1, N)
        attn_diffs.append(attn_diff.squeeze(0))  # (N,)

    # Aggregate over layers with weighted mean
    num_layers = len(attn_diffs)
    if num_layers == 0:
        raise ValueError("No attention maps collected")

    if num_layers == 1:
        weights = [1.0]
    elif num_layers == 2:
        weights = [0.3, 0.7]
    elif num_layers == 3:
        weights = [0.2, 0.3, 0.5]
    else:
        weights = torch.linspace(0.1, 0.9, num_layers).tolist()
        weights = [w / sum(weights) for w in weights]  # Normalize

    # Reshape each attention diff to spatial dimensions (H, W)
    # Each attn_diff has shape (N,) where N = H * W
    attn_spatials = []
    for attn_diff in attn_diffs:
        num_spatial = attn_diff.shape[0]
        # Find H, W such that H * W = num_spatial
        H = W = int(np.sqrt(num_spatial))
        if H * W != num_spatial:
            # If not perfect square, find closest factors
            H = int(np.sqrt(num_spatial))
            W = (num_spatial + H - 1) // H  # Ceiling division
        attn_spatial = attn_diff.view(H, W)
        attn_spatials.append(attn_spatial)

    # Interpolate all to a common size (use the largest spatial dimension)
    max_h = max(sp.shape[0] for sp in attn_spatials)
    max_w = max(sp.shape[1] for sp in attn_spatials)

    aggregated_attn = None
    for i, (attn_spatial, weight) in enumerate(zip(attn_spatials, weights)):
        # Interpolate to common size
        if attn_spatial.shape != (max_h, max_w):
            attn_spatial = F.interpolate(
                attn_spatial.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                size=(max_h, max_w),
                mode='bilinear',
                align_corners=False
            ).squeeze()  # (H, W)

        if aggregated_attn is None:
            aggregated_attn = weight * attn_spatial
        else:
            aggregated_attn += weight * attn_spatial

    mask = aggregated_attn
    max_v = (mask.mean() * clamp_rate).item()
    if max_v > 0:
        mask = mask.clamp(0, max_v) / max_v
    else:
        mask = torch.zeros_like(mask)

    mask = mask.detach().cpu().apply_(lambda pix: to_binary(pix, threshold))
    mask = mask.to(noisy_latent.device)
    _, _, latent_h, latent_w = noisy_latent.shape

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(
        mask,
        size=(latent_h, latent_w),
        mode='bilinear',
        align_corners=False
    )

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


def preprocess_image_for_vae(
    image: Image.Image,
    size: tuple = (512, 512),
    device: str = "cuda",
) -> torch.Tensor:
    """
    Preprocess PIL image for VAE encoding.
    Returns:
        Preprocessed tensor, shape (1, 3, H, W), values in [-1, 1] range
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    pixel_values = transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    return pixel_values