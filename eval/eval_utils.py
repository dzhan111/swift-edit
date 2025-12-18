import time
import os
from typing import Tuple

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


def mse_image(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute MSE between two images (H,W,3), values in [0,1]. If mask provided, compute over mask==1 region."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = (a - b) ** 2
    if mask is None:
        return float(np.mean(diff))
    mask = mask.astype(np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    denom = mask.mean()
    if denom == 0:
        return float('nan')
    return float((diff * mask).sum() / (mask.sum() * a.shape[2]))


def psnr_image(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None, data_range: float = 1.0) -> float:
    """Compute PSNR (dB) for images in [0,1]."""
    m = mse_image(a, b, mask=mask)
    if m == 0:
        return float('inf')
    if np.isnan(m):
        return float('nan')
    return 20.0 * float(np.log10(data_range)) - 10.0 * float(np.log10(m))


def upsample_mask_to_pixels(mask: torch.Tensor, target_size: Tuple[int, int], device: str = 'cpu') -> np.ndarray:
    """Upsample latent mask tensor (1,1,H_latent,W_latent) to pixel size (H_pix,W_pix).
    Returns binary numpy mask (H_pix,W_pix) with values 0/1.
    """
    if not isinstance(mask, torch.Tensor):
        raise ValueError('mask must be torch.Tensor')
    # move to device
    mask_t = mask.to(device)
    mask_up = F.interpolate(mask_t, size=(target_size[1], target_size[0]), mode='bilinear', align_corners=False)
    mask_up = mask_up.squeeze().detach().cpu().numpy()
    # ensure binary
    mask_bin = (mask_up > 0.5).astype(np.uint8)
    return mask_bin


def pil_to_numpy(img: Image.Image, size: Tuple[int, int] = None) -> np.ndarray:
    """Convert PIL Image to numpy float32 array in [0,1], optionally resize to size (W,H)."""
    if size is not None:
        img = img.resize(size, resample=Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def apply_mask_to_pil_region(img: Image.Image, mask_np: np.ndarray, background_color=(255, 255, 255)) -> Image.Image:
    """Return a PIL Image where only masked region is kept and background filled with background_color.
    mask_np expected shape (H,W) values 0/1.
    """
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    mask_f = mask_np.astype(np.float32)[..., None]
    bg = np.array(background_color, dtype=np.float32)
    comp = arr * mask_f + bg * (1.0 - mask_f)
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    return Image.fromarray(comp)


def compute_clip_similarity_whole(aux_models, img1: Image.Image, img2: Image.Image, device: str = 'cpu') -> float:
    """Compute cosine similarity between two PIL images using AuxiliaryModels.encode_image.
    Returns a scalar float in [-1,1]."""
    emb = aux_models.encode_image([img1, img2])
    # emb is torch.Tensor (2, dim)
    emb = emb.to(device)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    sim = torch.nn.functional.cosine_similarity(emb[0:1], emb[1:2], dim=-1)
    return float(sim.item())


def compute_clip_similarity_region(aux_models, src_img: Image.Image, edited_img: Image.Image, mask_np: np.ndarray, device: str = 'cpu') -> float:
    """Compute CLIP cosine similarity on masked regions (composited on white background).
    """
    src_reg = apply_mask_to_pil_region(src_img, mask_np, background_color=(255, 255, 255))
    edt_reg = apply_mask_to_pil_region(edited_img, mask_np, background_color=(255, 255, 255))
    return compute_clip_similarity_whole(aux_models, src_reg, edt_reg, device=device)
