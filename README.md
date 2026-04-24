# SVD-InST

SVD-InST is a style transfer project built on top of Stable Diffusion and latent diffusion components. This repository was created for the AI3603 course project at Shanghai Jiao Tong University.

![SVD-InST overview](./Images/SVD-InST.png)

Additional sample results:

![Result 1](./Images/result1.png)
![Result 2](./Images/result2.png)

For the full method description, see `Report.pdf`.

## Overview

The repository contains:

- training code for the SVD-InST variant of Stable Diffusion
- inference scripts for generating stylized images from content images
- configuration files for finetuning and inference
- example figures used in the project report

## Repository layout

```text
.
|-- configs/                # Training and inference configs
|-- ldm/                    # Latent diffusion modules
|-- logs/                   # Learned embeddings / model artifacts
|-- models/                 # Base Stable Diffusion checkpoints
|-- main.py                 # Main training entrypoint
|-- svdInST1.py             # Batch inference on a directory of images
|-- svdinst.py              # Single-image style transfer example
|-- InST.ipynb              # Notebook-based experimentation
`-- Report.pdf              # Project report
```

## Environment setup

Create the conda environment defined in `environment.yaml`:

```sh
conda env create -f environment.yaml
conda activate ldm
```

The environment targets:

- Python 3.8
- PyTorch 1.10
- CUDA 11.3

## Installation

Clone the repository and install the local package:

```sh
git clone https://github.com/dajiaohuang/SVD-InST.git
cd SVD-InST
pip install -e .
```

## Checkpoint preparation

This project expects both a base Stable Diffusion checkpoint and SVD-InST finetuned artifacts.

### 1. Base Stable Diffusion checkpoint

Download the original Stable Diffusion v1.4 checkpoint:

- https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt

Then place it at:

```text
models/sd/sd-v1-4.ckpt
```

If the `models/sd/` directory does not exist yet, create it first.

### 2. SVD-InST finetuned artifacts

The provided inference scripts expect:

```text
logs/embeddings.pt
logs/model.pt
```

At the moment, this repository already includes `logs/embeddings.pt`, but `logs/model.pt` is not included and must be supplied separately from your trained run or released checkpoint.

## Data preparation

Training uses `ldm.data.personalized.PersonalizedBase`, which reads all images directly from a single folder:

```text
/path/to/train_images
|-- image_001.jpg
|-- image_002.jpg
|-- image_003.png
`-- ...
```

Notes:

- images are resized to `512 x 512` during preprocessing
- captions are generated automatically with the placeholder token `*`
- the training command passes the image folder through `--data_root`

## Training

Run training with:

```sh
python main.py \
  --base configs/v1-finetune-svdiff.yaml \
  -t \
  --actual_resume models/sd/sd-v1-4.ckpt \
  -n <run_name> \
  --gpus 0, \
  --data_root /path/to/train_images
```

Useful related configs:

- `configs/v1-finetune-svdiff.yaml`: SVD-InST finetuning config
- `configs/stable-diffusion/v1-finetune.yaml`: baseline Stable Diffusion finetuning config

Training outputs are written under `logs/`.

## Inference

### Batch generation for a directory of content images

To generate multiple stylized outputs for every image in a directory:

```sh
python svdInST1.py /path/to/test_images
```

This script:

- loads `configs/stable-diffusion/v1-inference-svdiff.yaml`
- expects `models/sd/sd-v1-4.ckpt`
- expects `logs/embeddings.pt`
- expects `logs/model.pt`
- writes results under `outputs/img2img-samples/`

### Single-image example

For a single content image workflow, use `svdinst.py`.

Before running it, review the hard-coded paths inside the script and update them for your machine, especially:

- the content image path
- the style image path
- the checkpoint locations

### Notebook workflow

If you prefer an interactive workflow, `InST.ipynb` contains a notebook version of the pipeline.

## Known limitations

- Some inference scripts still contain hard-coded local paths from the original project environment.
- The README documents the expected file layout, but you may still need to adjust script paths before running inference on a new machine.
- Only part of the trained artifacts are stored in this repository.

## Demo and report

- Report: `Report.pdf`
- Demo video: https://www.bilibili.com/video/BV1hg4y1r7z4

## License

See `LICENSE` for the repository license.

