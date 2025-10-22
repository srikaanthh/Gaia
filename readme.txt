# Land-Cover Semantic Segmentation (U-Net + EfficientNet-B0)

A production-ready PyTorch pipeline for land-cover semantic segmentation on high-resolution satellite imagery. It reproduces and extends ideas from “Segmenting the Earth: Challenges in Land Cover Classification” (Shirley Cheng, Stanford, 2023) and adds practical engineering for local training/inference, dataset patching, metrics, and performance.

## Key features
- U-Net backbone with EfficientNet-B0 (via Segmentation Models PyTorch)
- Patch-based training/inference for large imagery (configurable tile size)
- Multiple datasets: LandCover.ai (native) + optional DeepGlobe conversion/merge
- Robust train/val split with reproducibility
- Clean dataset class with vectorized mask handling (fast and memory-friendly)
- Batched inference over patches (significant speedup)
- Logging to disk, best-model checkpointing

See OPTIMIZATION_SUMMARY.md for details on the performance improvements (num_workers, cuDNN, batched patch inference, etc.).

---

## Repository structure

```
Land-Cover-Semantic-Segmentation-PyTorch-main/
├─ assets/                      # images used in docs
├─ config/
│  └─ config.yaml              # central configuration for scripts
├─ data/                       # datasets (created/downloaded locally)
│  ├─ train/
│  │  ├─ images/
│  │  └─ masks/
│  ├─ test/
│  │  ├─ images/
│  │  └─ masks/
│  └─ patches_512/            # generated patches and splits
├─ models/                     # saved trained weights (.pth)
├─ notebooks/
│  ├─ training.ipynb          # original Colab-centric notebook
│  └─ training 2.ipynb        # local-friendly; supports LandCover + DeepGlobe merge
├─ output/                    # predictions, plots
├─ src/
│  ├─ train.py                # training entrypoint (config-driven)
│  ├─ test.py                 # inference + plots (config-driven)
│  └─ utils/
│     ├─ constants.py
│     ├─ dataset.py           # optimized dataset loader
│     ├─ logger.py
│     ├─ patching.py          # image/mask patchification utilities
│     ├─ plot.py
│     ├─ preprocess.py        # augment/preprocess pipelines
│     └─ root_config.py       # config loader and path roots
├─ requirements.txt
├─ OPTIMIZATION_SUMMARY.md
└─ README.md (this file)
```

---

## Datasets

This project supports:
- LandCover.ai (native)
- DeepGlobe Land Cover Classification (optional; converted to LandCover.ai label space)

Masks are expected as single-channel integer class IDs. The default classes are:

```
0 = background
1 = building
2 = woodland
3 = water
4 = road (optional; can be excluded in config)
```

### LandCover.ai (Kaggle)
- The local-friendly notebook `notebooks/training 2.ipynb` includes a Python cell to unzip `landcoverai.zip` into `data/`.
- Alternatively, download with Kaggle CLI (see Setup below) and unzip to `data/`.

### DeepGlobe (optional fusion)
- `notebooks/training 2.ipynb` contains cells to download DeepGlobe from Kaggle, convert masks to LandCover.ai-compatible IDs, patchify, and merge patches with LandCover.ai before splitting.
- You may need to adjust the mapping dictionary `DG_TO_LCAI` depending on your specific DeepGlobe distribution (ID/palette). Verify via `np.unique(mask)` and update accordingly.

---

## Setup (macOS, zsh)

Prerequisites:
- Python 3.9+ (3.10 recommended)
- CUDA GPU optional but recommended (Linux). On macOS you can train on CPU or Apple Silicon (adjust device in config).
- Kaggle account + API token (kaggle.json)

1) Create and activate a virtual environment (recommended)

```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install Python dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

3) Kaggle CLI credentials

- Download your Kaggle API token from https://www.kaggle.com/ > Account > Create New API Token
- Save as `~/.kaggle/kaggle.json` with correct permissions:

```
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

4) Download datasets (optional from terminal)

```
# LandCover.ai
kaggle datasets download -d adrianboguszewski/landcoverai -p data
unzip -q data/landcoverai.zip -d data

# DeepGlobe Land Cover (example slug)
kaggle datasets download -d zfturbo/deepglobe-land-cover-classification-dataset -p data
unzip -q data/*.zip -d data/deepglobe
```

Note: The notebook `training 2.ipynb` can also perform these via Python cells.

---

## Configuration

All runtime options are managed via `config/config.yaml` and `utils/constants.py`. Key knobs:

- vars:
	- patch_size: size of training tiles (e.g., 512)
	- batch_size: training batch size (e.g., 16)
	- model_arch: U-Net variant from SMP (e.g., Unet)
	- encoder: EfficientNet-B0 (default) or other SMP encoders
	- encoder_weights: imagenet
	- activation: softmax2d or None
	- optimizer_choice: Adam, SGD, etc.
	- init_lr: initial learning rate
	- epochs: number of epochs
	- train_classes / test_classes: subset of classes to include
	- device: cuda or cpu
- dirs:
	- data_dir, train_dir, test_dir, image_dir, mask_dir
	- model_dir, output_dir, log_dir

The scripts read config and honor your local paths.

---

## Training

Use the optimized Python script:

```
python src/train.py
```

What it does:
- Patchifies images/masks into `data/train/patches_{patch_size}`
- Discards background-dominant patches based on `discard_rate`
- Splits into train/val (default 80/20)
- Builds U-Net+EfficientNet-B0, trains, logs IoU/Dice
- Saves best model to `models/` (or as configured)
- Preserves patches by default for faster re-runs

Performance tips:
- DataLoader uses tuned `num_workers`, `pin_memory`, `persistent_workers`
- cuDNN benchmark enabled on CUDA
- Consider enabling mixed precision (AMP) with a custom loop if you need more speed

---

## Inference & evaluation

Run batched patch-based inference:

```
python src/test.py
```

What it does:
- Loads best checkpoint (configurable)
- Predicts masks in batched patches (fast)
- Reconstructs full-size masks, saves to `output/.../predicted_masks/`
- Saves side-by-side plots to `output/.../prediction_plots/`

---

## Notebooks

- `notebooks/training 2.ipynb` is adapted for local use (macOS/Linux). It can:
	- Download & unzip LandCover.ai
	- Download DeepGlobe, convert masks to LandCover.ai IDs, patchify, and merge
	- Visualize samples
	- Optionally train/eval within the notebook (or call into the scripts)

Avoid Colab-only cells (apt-get, /content drive mount) on local runs.

---

## Results (example)

- Backbone: EfficientNet-B0, Unet
- Tile: 512×512, batch 16, LR 3e-4
- Metrics: IoU@0.5, Dice loss

Your exact numbers depend on dataset mix, mapping quality (DeepGlobe→LandCover), and training time. Use the logs in `logs/` and saved best models in `models/` for comparison.

---

## Troubleshooting

- Kaggle: “401/403” → ensure `~/.kaggle/kaggle.json` exists with `chmod 600`.
- CUDA OOM: reduce `batch_size` or `patch_size`.
- Slow dataloading: increase `num_workers` in `src/train.py` or ensure dataset resides on fast storage.
- Class mismatch: verify mask unique values and adjust `DG_TO_LCAI` mapping (DeepGlobe section in notebook).
- Seam artifacts at inference: either keep overlap-and-blend or increase patch size.

---

## Citation

Please cite the original datasets and SMP library if you use this in academic work. Consider referencing:
- Segmentation Models PyTorch (SMP): https://github.com/qubvel/segmentation_models.pytorch
- LandCover.ai: https://landcover.ai
- DeepGlobe 2018: https://deepglobe.org
- Cheng, S. “Segmenting the Earth: Challenges in Land Cover Classification,” Stanford (2023)

---

## License

This repository is released under the LICENSE included in this repo. Datasets have their own licenses—please review and comply with each dataset’s terms when downloading and using them.

---

## Author

Srikanth Akkaru  
Master’s in Computer Science, University of South Florida  
akkarusrikanth@gmail.com  
GitHub: https://github.com/srikaanthh