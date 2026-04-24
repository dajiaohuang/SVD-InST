# SVD-InST

SVD-InST is a style transfer project built on top of Stable Diffusion. The method combines the image-conditioned personalization idea from InST with the singular-value finetuning strategy of SVDiff, and was developed as the final project for AI3603 at Shanghai Jiao Tong University.

![SVD-InST overview](./Images/SVD-InST.png)

Additional sample results:

![Result 1](./Images/result1.png)
![Result 2](./Images/result2.png)

For the original project report and presentation material, see `17_苏展_吴舒文_运嘉盛.pdf` and `Report.pdf`.

## Overview

The repository contains:

- training code for the SVD-InST variant of Stable Diffusion
- inference scripts for generating stylized images from content images
- configuration files for finetuning and inference
- example figures used in the project report

## Method summary

According to the project report, SVD-InST is designed to improve style transfer quality by combining two ideas:

1. **InST-style conditioning**: use a learned placeholder token and image-aware conditioning to inject style information into the text/image generation pipeline.
2. **SVDiff-style parameterization**: decompose pretrained weights with SVD and optimize lightweight singular-value updates instead of fully finetuning the whole diffusion model.

In this repository, that design is reflected by:

- `ldm.modules.embedding_manager.EmbeddingManager`, which learns the style-related embedding used with the placeholder token `*`
- `SVDmodel.py`, which defines SVD-based layers such as `SVDConv2d` and `SVDLinear`
- `configs/v1-finetune-svdiff.yaml`, which switches the UNet to the SVD-based implementation for finetuning

The goal is to keep the strong generation prior of Stable Diffusion while improving style faithfulness and preserving content structure.

## Experimental results

The report compares SVD-InST against several style transfer baselines, including InST, CycleGAN, StyTR-2, and fast-style-transfer.

![Quantitative and qualitative comparison](./Images/comparison_data.png)

Reported quantitative results from the project report:

| Model | FID | LPIPS |
| --- | ---: | ---: |
| InST | 127.5 | 0.53 |
| CycleGAN | 178.3 | - |
| StyTR-2 | 171.3 | - |
| fast-style-transfer | 172.7 | - |
| SVD-InST (ours) | **125.1** | **0.54** |

Lower FID indicates better distributional quality. The report shows that SVD-InST achieves the best FID among the compared methods while keeping LPIPS at a level similar to or slightly better than InST.

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
- the project follows the personalized-style setup from the report, so a small folder of style-domain images is sufficient for finetuning

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

During training, the code saves the learned textual embedding and the finetuned diffusion weights separately. In practice, the artifacts you will usually need later are:

```text
logs/embeddings.pt
logs/model.pt
```

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

The batch script generates multiple outputs for each input content image, matching the evaluation workflow described in the report.

### Single-image example

For a single content image workflow, use `svdinst.py`.

Before running it, review the hard-coded paths inside the script and update them for your machine, especially:

- the content image path
- the style image path
- the checkpoint locations

At inference time, the pipeline loads the embedding checkpoint, applies `perform_svd()` to the SVD-based modules, and then restores the finetuned model weights before sampling.

### Notebook workflow

If you prefer an interactive workflow, `InST.ipynb` contains a notebook version of the pipeline.

## Known limitations

- Some inference scripts still contain hard-coded local paths from the original project environment.
- The README documents the expected file layout, but you may still need to adjust script paths before running inference on a new machine.
- Only part of the trained artifacts are stored in this repository.

## Demo and report

- Project report: `17_苏展_吴舒文_运嘉盛.pdf`
- Additional report file: `Report.pdf`
- Demo video: https://www.bilibili.com/video/BV1hg4y1r7z4

## License

See `LICENSE` for the repository license.

