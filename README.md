# Usage Guide

### Introduction

This is a reimplementation of Swiftedit based on the paper

### Step 1.1: Create Python Environment

**Option A: Using Conda (Recommended)**
```bash
# Create new environment
conda create -n swiftedit python=3.10 -y
conda activate swiftedit
```

### Step 1.2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Downloading helper model weights

### Step 2.1: Understanding Required Models

SwiftEdit relies upon the following (pretrained) weights:

1. **Inversion Model** (`inverse_ckpt-120k ("F_θ" in the paper)`) - Converts images to noise
2. **SwiftBrushV2 Model** (`sbv2_0.5`) - One-step diffusion generator
3. **IP-Adapter** - Image conditioning

### Step 2.2: Download Base Models

Run the following:

```bash

curl -L -O  https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-aa
curl -L -O  https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ab
curl -L -O  https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ac
curl -L -O  https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ad
curl -L -O  https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ae
```

### Step 2.3: Extract Weights

```bash
cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz

tar zxvf swiftedit_weights.tar.gz

# these files are large so remove to save space
rm swiftedit_weights.tar.gz.part-*
rm swiftedit_weights.tar.gz
```

## 3. Running our implementation

### Step 3.1: Prepare Your Image

```bash
# Create test directory and place images in there
mkdir -p test_images
# Place your source images in test_images/
```

### Step 3.2: Run test_edit.py

**Basic Usage:**
```bash
python test_edit.py
```

**With Custom Arguments:**
```bash
python test_edit.py \
    --source_image_path test_images/your_image.jpg \
    --source_prompt "a cat" \
    --edit_prompt "a dog" \
    --device cuda \
    --mask_method attention
```

### Step 3.3: Available Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source_image_path` | str | `test_images/woman_face.jpg` | Path to source image |
| `--source_prompt` | str | `"woman"` | Description of the source image |
| `--edit_prompt` | str | `"taylor swift"` | Description of desired edit |
| `--output_image_path` | str | `None` | Output path (auto-generated if not specified) |
| `--device` | str | `"cpu"` | Device to run on: `"cpu"` or `"cuda"` |
| `--dtype` | str | `"fp32"` | Weight dtype: `"fp32"`, `"fp16"`, or `"bf16"` |
| `--mask_method` | str | `"noise"` | Mask extraction method: `"noise"` or `"attention"` |

**Note:** If `--output_image_path` is not specified, the output will be saved as `{source_name}_edited_{mask_method}.{ext}` in the same directory as the source image.

### Step 3.4: Mask Extraction Methods

This implementation supports two mask extraction methods:

1. **Noise-based (`--mask_method noise`)** - Default method
   - Uses differences in noise predictions from the inversion model
   - Computes mask at mid-timestep (t=500)
   - More global, based on noise prediction differences

2. **Attention-based (`--mask_method attention`)** - New method
   - Uses cross-attention maps from the SBv2 generator
   - Captures where the model attends differently between prompts
   - More semantically focused, aligned with model attention

**Example: Compare both methods**
```bash
# Noise-based mask
python test_edit.py --mask_method noise --source_image_path test_images/apples.jpg \
    --source_prompt "basket of apples" --edit_prompt "basket of puppies"

# Attention-based mask
python test_edit.py --mask_method attention --source_image_path test_images/apples.jpg \
    --source_prompt "basket of apples" --edit_prompt "basket of puppies"
```

The outputs will be saved as `apples_edited_noise.jpg` and `apples_edited_attention.jpg` respectively.

### NOTES:

All weights are trained and provided by the SwiftEdit authors (Qualcomm AI Research).

- SwiftEdit GitHub: https://github.com/Qualcomm-AI-research/SwiftEdit
- SwiftEdit Paper: https://swift-edit.github.io/

## License

The weights are subject to the BSD-3-Clause license from the SwiftEdit repository.