from inference.edit_image import SwiftEditPipeline
import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="SwiftEdit Test Script - Quick test of SwiftEdit pipeline"
    )
    parser.add_argument(
        "--source_image_path",
        type=str,
        default="test_images/woman_face.jpg",
        help="Path to source image (default: test_images/woman_face.jpg)",
    )
    parser.add_argument(
        "--source_prompt",
        type=str,
        default="woman",
        help="Source description (default: 'woman')",
    )
    parser.add_argument(
        "--edit_prompt",
        type=str,
        default="taylor swift",
        help="Edit description (default: 'taylor swift')",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default=None,
        help="Output image path (default: auto-generated from input)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Weight dtype (default: fp32)",
    )
    parser.add_argument(
        "--mask_method",
        type=str,
        default="noise",
        choices=["noise", "attention"],
        help="Mask extraction method: 'noise' (default) or 'attention'",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    weights_root = os.path.join(os.path.dirname(__file__), "swiftedit_weights")
    inversion_path = os.path.join(weights_root, "inverse_ckpt-120k")
    sbv2_path = os.path.join(weights_root, "sbv2_0.5")
    ip_adapter_path = os.path.join(
        weights_root, "ip_adapter_ckpt-90k", "ip_adapter.bin"
    )

    if args.output_image_path is None:
        source_path = Path(args.source_image_path)
        output_path = source_path.parent / f"{source_path.stem}_edited_{args.mask_method}{source_path.suffix}"
    else:
        output_path = Path(args.output_image_path)

    print("Loading SwiftEdit pipeline...")
    pipeline = SwiftEditPipeline(
        inversion_model_path=inversion_path,
        sbv2_model_path=sbv2_path,
        ip_adapter_path=ip_adapter_path,
        device=args.device,
        dtype=args.dtype,
    )

    edited_image = pipeline.edit(
        source_image=args.source_image_path,
        source_prompt=args.source_prompt,
        edit_prompt=args.edit_prompt,
        mask_method=args.mask_method,
    )

    edited_image.save(str(output_path))
    print("Edited image saved to", output_path)

if __name__ == "__main__":
    main()
