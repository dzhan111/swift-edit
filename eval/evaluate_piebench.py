import argparse
import os
import sys
import time
import csv
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
import numpy as np
from PIL import Image

from inference.edit_image import SwiftEditPipeline
from utils.editing_utils import (
    extract_editing_mask,
    extract_attention_based_mask,
    postprocess_image,
)
from eval.eval_utils import (
    upsample_mask_to_pixels,
    pil_to_numpy,
    mse_image,
    psnr_image,
    compute_clip_similarity_whole,
    compute_clip_similarity_region,
)


def sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    return time.perf_counter()


@torch.no_grad()
def run_pie_evaluation(
    dataset_config,
    output_dir,
    methods,
    inversion_model_path,
    sbv2_model_path,
    ip_adapter_path,
    device,
    dtype,
    csv_path,
    threshold=0.5,
    max_samples=None,
    clip_scale=100.0,
    mse_scale=1e4,
    time_mode="end2end",  # "end2end" or "gen_only"
    save_outputs=True,
    compute_metrics=True,
    warmup=0,
):
    """
    Evaluate SwiftEdit on PIE_Bench_pp dataset.

    clip_scale: multiply CLIP similarities by this factor (e.g., 100 to match many papers)
    mse_scale: multiply MSE by this factor (e.g., 1e4 if you report MSE x 10^4)
    time_mode:
      - "end2end": encode + invert + mask + generate (excludes optional saving/metrics)
      - "gen_only": time only the generate() call (paper-style latency)
    save_outputs: whether to save edited images/masks
    compute_metrics: whether to compute PSNR/MSE/CLIP
    warmup: number of warmup iterations (not recorded) to stabilize timing
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(f"Loading PIE_Bench_pp with config: {dataset_config}")
    dataset = load_dataset("UB-CVML-Group/PIE_Bench_pp", dataset_config)
    dataset = dataset["V1"]

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Dataset loaded: {len(dataset)} samples")

    pipeline = SwiftEditPipeline(
        inversion_model_path=inversion_model_path,
        sbv2_model_path=sbv2_model_path,
        ip_adapter_path=ip_adapter_path,
        device=device,
        dtype=dtype,
    )

    # Choose autocast based on dtype
    use_autocast = (device.startswith("cuda") and dtype.lower() in ("fp16", "float16"))
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else nullcontext()

    fieldnames = [
        "sample_id", "edit_action", "method",
        "psnr_bg", "mse_bg", "clip_whole", "clip_edited",
        "time_encode", "time_invert", "time_mask", "time_gen",
        "time_total",
        "time_mode", "clip_scale", "mse_scale",
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Optional warmup on first sample/method to stabilize CUDA kernels
        warmup_done = 0

        for idx, sample in enumerate(dataset):
            sample_id = sample["id"]
            src_img = sample["image"].convert("RGB") if sample["image"].mode != "RGB" else sample["image"]
            source_prompt = sample["source_prompt"]
            target_prompt = sample["target_prompt"]
            edit_action = sample.get("edit_action", "unknown")

            print(f"\n[{idx+1}/{len(dataset)}] Processing {sample_id}")
            print(f"  Action: {edit_action}")
            print(f"  Source: '{source_prompt}' -> Target: '{target_prompt}'")

            for method in methods:
                try:
                    # Warmup loop (not recorded)
                    if warmup_done < warmup:
                        with autocast_ctx:
                            image_latent = pipeline.inversion_model.encode_image(src_img)
                            inverted_noise = pipeline.inversion_model.invert(image_latent, prompt=source_prompt)
                            noisy_latent = (
                                pipeline.ip_sbv2_model.alpha_t * image_latent +
                                pipeline.ip_sbv2_model.sigma_t * inverted_noise
                            )
                            if method == "attention":
                                edit_mask = extract_attention_based_mask(
                                    ip_sbv2_model=pipeline.ip_sbv2_model,
                                    noisy_latent=noisy_latent,
                                    source_prompt=source_prompt,
                                    edit_prompt=target_prompt,
                                    source_image=src_img,
                                    threshold=threshold,
                                )
                            else:
                                edit_mask = extract_editing_mask(
                                    inversion_model=pipeline.inversion_model,
                                    image_latent=image_latent,
                                    source_prompt=source_prompt,
                                    edit_prompt=target_prompt,
                                    threshold=threshold,
                                )
                            from models.mask_controller import MaskController
                            mask_controller = MaskController(mask=edit_mask, scale_text=1.0, scale_edit=0.2, scale_non_edit=1.0)
                            pipeline.ip_sbv2_model.set_mask_controller(mask_controller, where=["mid_blocks", "up_blocks"])
                            _ = pipeline.ip_sbv2_model.generate(
                                inverted_noise=noisy_latent,
                                prompts=[source_prompt, target_prompt],
                                source_image=src_img,
                                scale=1.0,
                            )
                        sync(device)
                        warmup_done += 1
                        continue

                    # --------------------------
                    # Timing block
                    # --------------------------
                    time_encode = time_invert = time_mask = time_gen = 0.0

                    if time_mode == "end2end":
                        total_start = now()

                    # Step 1: encode
                    enc_start = now()
                    sync(device)
                    with autocast_ctx:
                        image_latent = pipeline.inversion_model.encode_image(src_img)
                    sync(device)
                    time_encode = now() - enc_start

                    # Step 2: invert
                    inv_start = now()
                    sync(device)
                    with autocast_ctx:
                        inverted_noise = pipeline.inversion_model.invert(image_latent, prompt=source_prompt)
                    sync(device)
                    time_invert = now() - inv_start

                    noisy_latent = (
                        pipeline.ip_sbv2_model.alpha_t * image_latent +
                        pipeline.ip_sbv2_model.sigma_t * inverted_noise
                    )

                    # Step 3: mask extraction
                    mask_start = now()
                    sync(device)
                    with autocast_ctx:
                        if method == "attention":
                            edit_mask = extract_attention_based_mask(
                                ip_sbv2_model=pipeline.ip_sbv2_model,
                                noisy_latent=noisy_latent,
                                source_prompt=source_prompt,
                                edit_prompt=target_prompt,
                                source_image=src_img,
                                threshold=threshold,
                            )
                        else:
                            edit_mask = extract_editing_mask(
                                inversion_model=pipeline.inversion_model,
                                image_latent=image_latent,
                                source_prompt=source_prompt,
                                edit_prompt=target_prompt,
                                threshold=threshold,
                            )
                    sync(device)
                    time_mask = now() - mask_start

                    # Step 4: generation
                    from models.mask_controller import MaskController
                    mask_controller = MaskController(mask=edit_mask, scale_text=1.0, scale_edit=0.2, scale_non_edit=1.0)
                    pipeline.ip_sbv2_model.set_mask_controller(mask_controller, where=["mid_blocks", "up_blocks"])

                    gen_start = now()
                    sync(device)
                    with autocast_ctx:
                        images_tensor, _ = pipeline.ip_sbv2_model.generate(
                            inverted_noise=noisy_latent,
                            prompts=[source_prompt, target_prompt],
                            source_image=src_img,
                            scale=1.0,
                        )
                    sync(device)
                    time_gen = now() - gen_start

                    if time_mode == "gen_only":
                        time_total = time_gen
                    else:
                        time_total = now() - total_start

                    # Postprocess edited image
                    edited_pil = postprocess_image(images_tensor[1:2])

                    # Resize source to edited size for fair comparison
                    tgt_size = edited_pil.size
                    src_resized = src_img.resize(tgt_size, resample=Image.LANCZOS)

                    # Upsample mask to pixel size
                    mask_np = upsample_mask_to_pixels(edit_mask, target_size=tgt_size, device=pipeline.device)

                    # Optionally save outputs (excluded from timing)
                    if save_outputs:
                        out_base = os.path.join(output_dir, f"{sample_id}_{method}")
                        edited_pil.save(out_base + "_edited.png")
                        Image.fromarray((mask_np * 255).astype("uint8")).save(out_base + "_mask.png")

                    # Optionally compute metrics (excluded from timing)
                    psnr_bg = mse_bg = clip_whole = clip_edited = None
                    if compute_metrics:
                        src_np = pil_to_numpy(src_resized)
                        edt_np = pil_to_numpy(edited_pil)

                        bg_mask = 1 - mask_np
                        mse_bg = mse_image(src_np, edt_np, mask=bg_mask) * float(mse_scale)
                        psnr_bg = psnr_image(src_np, edt_np, mask=bg_mask)

                        clip_whole = compute_clip_similarity_whole(
                            pipeline.aux_models, src_resized, edited_pil, device=pipeline.device
                        ) * float(clip_scale)
                        clip_edited = compute_clip_similarity_region(
                            pipeline.aux_models, src_resized, edited_pil, mask_np, device=pipeline.device
                        ) * float(clip_scale)

                    row = {
                        "sample_id": sample_id,
                        "edit_action": edit_action,
                        "method": method,
                        "psnr_bg": psnr_bg,
                        "mse_bg": mse_bg,
                        "clip_whole": clip_whole,
                        "clip_edited": clip_edited,
                        "time_encode": time_encode,
                        "time_invert": time_invert,
                        "time_mask": time_mask,
                        "time_gen": time_gen,
                        "time_total": time_total,
                        "time_mode": time_mode,
                        "clip_scale": clip_scale,
                        "mse_scale": mse_scale,
                    }

                    writer.writerow(row)
                    print(
                        f"  [{method}] psnr_bg={psnr_bg:.3f} "
                        f"clip_whole={clip_whole:.3f} "
                        f"time_total={time_total:.3f}s (mode={time_mode}) ✓"
                    )

                except Exception as e:
                    print(f"  [{method}] ERROR: {str(e)}")
                    continue


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SwiftEdit on PIE_Bench_pp dataset")

    p.add_argument("--dataset_config", type=str, default="0_random_140",
                   help="PIE_Bench_pp config (e.g., 0_random_140, 1_change_object_80, etc.)")
    p.add_argument("--methods", type=str, default="noise,attention",
                   help="Comma-separated mask extraction methods")
    p.add_argument("--output_dir", type=str, default="eval_pie_outputs",
                   help="Directory to save outputs")
    p.add_argument("--inversion_model_path", type=str, default="swiftedit_weights/inverse_ckpt-120k")
    p.add_argument("--sbv2_model_path", type=str, default="swiftedit_weights/sbv2_0.5")
    p.add_argument("--ip_adapter_path", type=str, default="swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16")
    p.add_argument("--csv_path", type=str, default="eval_pie_results/results.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Maximum number of samples to evaluate (None = all)")

    # ---- New flags (for metric comparability) ----
    p.add_argument("--clip_scale", type=float, default=100.0,
                   help="Scale factor applied to CLIP similarities (e.g., 100.0 to match many paper tables).")
    p.add_argument("--mse_scale", type=float, default=1e4,
                   help="Scale factor applied to MSE (e.g., 1e4 if reporting MSE x 10^4).")
    p.add_argument("--time_mode", type=str, default="end2end", choices=["end2end", "gen_only"],
                   help="Timing mode: end2end (encode+invert+mask+gen) or gen_only (paper-style).")
    p.add_argument("--no_save", action="store_true",
                   help="Disable saving edited images and masks.")
    p.add_argument("--no_metrics", action="store_true",
                   help="Disable metric computation (timing only).")
    p.add_argument("--warmup", type=int, default=0,
                   help="Number of warmup iterations (not recorded) to stabilize CUDA timing.")

    return p.parse_args()


# Small helper to avoid importing contextlib everywhere
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False


if __name__ == "__main__":
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    run_pie_evaluation(
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        methods=methods,
        inversion_model_path=args.inversion_model_path,
        sbv2_model_path=args.sbv2_model_path,
        ip_adapter_path=args.ip_adapter_path,
        device=args.device,
        dtype=args.dtype,
        csv_path=args.csv_path,
        threshold=args.threshold,
        max_samples=args.max_samples,
        clip_scale=args.clip_scale,
        mse_scale=args.mse_scale,
        time_mode=args.time_mode,
        save_outputs=not args.no_save,
        compute_metrics=not args.no_metrics,
        warmup=args.warmup,
    )

    print(f"\nEvaluation complete! Results saved to {args.csv_path}")
