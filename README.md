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
pip install -r swift_edit/requirements.txt
```

### Step 1.3: Verify Installation

```bash
python -c "import swift_edit; print('SwiftEdit version:', swift_edit.__version__)"

# Should output: SwiftEdit version: 1.0.0
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
```

### Step 3.2: Run test_edit.py

```bash
python test_edit.py
```

### NOTES: 

All weights are trained and provided by the SwiftEdit authors (Qualcomm AI Research).

- SwiftEdit GitHub: https://github.com/Qualcomm-AI-research/SwiftEdit
- SwiftEdit Paper: https://swift-edit.github.io/

## License

The weights are subject to the BSD-3-Clause license from the SwiftEdit repository.