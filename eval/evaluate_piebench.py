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
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


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
):
    """
    Evaluate SwiftEdit on PIE_Bench_pp dataset.
    
    Args:
        dataset_config: Dataset config name (e.g., '0_random_140')
        output_dir: Directory to save edited images and masks
        methods: List of mask extraction methods to evaluate
        inversion_model_path: Path to inversion model
        sbv2_model_path: Path to stable diffusion v2 model
        ip_adapter_path: Path to IP-Adapter weights
        device: Device to run on ('cuda' or 'cpu')
        dtype: Data type ('fp16' or 'fp32')
        csv_path: Path to save CSV results
        threshold: Mask threshold
        max_samples: Maximum number of samples to evaluate (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Load dataset
    print(f"Loading PIE_Bench_pp with config: {dataset_config}")
    dataset = load_dataset("UB-CVML-Group/PIE_Bench_pp", dataset_config)
    dataset = dataset['V1']  # Select the V1 split
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Initialize pipeline
    pipeline = SwiftEditPipeline(
        inversion_model_path=inversion_model_path,
        sbv2_model_path=sbv2_model_path,
        ip_adapter_path=ip_adapter_path,
        device=device,
        dtype=dtype,
    )

    fieldnames = [
        'sample_id', 'edit_action', 'method', 'psnr_bg', 'mse_bg', 
        'clip_whole', 'clip_edited', 'time_encode', 'time_invert', 
        'time_mask', 'time_gen', 'time_total',
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, sample in enumerate(dataset):
            sample_id = sample['id']
            src_img = sample['image'].convert('RGB') if sample['image'].mode != 'RGB' else sample['image']
            source_prompt = sample['source_prompt']
            target_prompt = sample['target_prompt']
            edit_action = sample.get('edit_action', 'unknown')
            
            print(f"\n[{idx+1}/{len(dataset)}] Processing {sample_id}")
            print(f"  Action: {edit_action}")
            print(f"  Source: '{source_prompt}' -> Target: '{target_prompt}'")

            for method in methods:
                try:
                    total_start = time.perf_counter()

                    # Step 1: encode
                    enc_start = time.perf_counter()
                    sync(device)
                    image_latent = pipeline.inversion_model.encode_image(src_img)
                    sync(device)
                    enc_time = time.perf_counter() - enc_start

                    # Step 2: invert
                    inv_start = time.perf_counter()
                    sync(device)
                    inverted_noise = pipeline.inversion_model.invert(image_latent, prompt=source_prompt)
                    sync(device)
                    inv_time = time.perf_counter() - inv_start

                    # Construct noisy latent
                    noisy_latent = (
                        pipeline.ip_sbv2_model.alpha_t * image_latent +
                        pipeline.ip_sbv2_model.sigma_t * inverted_noise
                    )

                    # Step 3: mask extraction
                    mask_start = time.perf_counter()
                    sync(device)
                    if method == 'attention':
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
                    mask_time = time.perf_counter() - mask_start

                    # Step 4: generation
                    gen_start = time.perf_counter()
                    sync(device)
                    from models.mask_controller import MaskController

                    mask_controller = MaskController(mask=edit_mask, scale_text=1.0, scale_edit=0.2, scale_non_edit=1.0)
                    pipeline.ip_sbv2_model.set_mask_controller(mask_controller, where=['mid_blocks', 'up_blocks'])

                    images_tensor, _ = pipeline.ip_sbv2_model.generate(
                        inverted_noise=noisy_latent,
                        prompts=[source_prompt, target_prompt],
                        source_image=src_img,
                        scale=1.0,
                    )
                    sync(device)
                    gen_time = time.perf_counter() - gen_start

                    total_time = time.perf_counter() - total_start

                    # postprocess edited image
                    edited_pil = postprocess_image(images_tensor[1:2])

                    # resize source to edited size for fair comparison
                    tgt_size = edited_pil.size  # (W,H)
                    src_resized = src_img.resize(tgt_size, resample=Image.LANCZOS)

                    # upsample mask to pixel size
                    mask_np = upsample_mask_to_pixels(edit_mask, target_size=tgt_size, device=pipeline.device)

                    # Save outputs
                    out_base = os.path.join(output_dir, f"{sample_id}_{method}")
                    edited_pil.save(out_base + '_edited.png')
                    Image.fromarray((mask_np * 255).astype('uint8')).save(out_base + '_mask.png')

                    # compute metrics
                    src_np = pil_to_numpy(src_resized)
                    edt_np = pil_to_numpy(edited_pil)

                    bg_mask = 1 - mask_np
                    mse_bg = mse_image(src_np, edt_np, mask=bg_mask)
                    psnr_bg = psnr_image(src_np, edt_np, mask=bg_mask)

                    clip_whole = compute_clip_similarity_whole(pipeline.aux_models, src_resized, edited_pil, device=pipeline.device)
                    clip_edited = compute_clip_similarity_region(pipeline.aux_models, src_resized, edited_pil, mask_np, device=pipeline.device)

                    row = {
                        'sample_id': sample_id,
                        'edit_action': edit_action,
                        'method': method,
                        'psnr_bg': psnr_bg,
                        'mse_bg': mse_bg,
                        'clip_whole': clip_whole,
                        'clip_edited': clip_edited,
                        'time_encode': enc_time,
                        'time_invert': inv_time,
                        'time_mask': mask_time,
                        'time_gen': gen_time,
                        'time_total': total_time,
                    }

                    writer.writerow(row)
                    print(f"  [{method}] psnr_bg={psnr_bg:.3f}, clip_whole={clip_whole:.3f}, time={total_time:.2f}s ✓")
                
                except Exception as e:
                    print(f"  [{method}] ERROR: {str(e)}")
                    continue


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate SwiftEdit on PIE_Bench_pp dataset')
    p.add_argument('--dataset_config', type=str, default='0_random_140',
                   help='PIE_Bench_pp config (e.g., 0_random_140, 1_change_object_80, etc.)')
    p.add_argument('--methods', type=str, default='noise,attention',
                   help='Comma-separated mask extraction methods')
    p.add_argument('--output_dir', type=str, default='eval_pie_outputs',
                   help='Directory to save outputs')
    p.add_argument('--inversion_model_path', type=str, default='swiftedit_weights/inverse_ckpt-120k')
    p.add_argument('--sbv2_model_path', type=str, default='swiftedit_weights/sbv2_0.5')
    p.add_argument('--ip_adapter_path', type=str, default='swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--dtype', type=str, default='fp16')
    p.add_argument('--csv_path', type=str, default='eval_pie_results/results.csv')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--max_samples', type=int, default=None,
                   help='Maximum number of samples to evaluate (None = all)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    
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
    )
    
    print(f"\nEvaluation complete! Results saved to {args.csv_path}")
