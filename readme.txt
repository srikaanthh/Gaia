# Gaia 
### Land Cover Semantic Segmentation using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/Model-U--Net%20%2B%20EfficientNet-green" alt="Model">
  <img src="https://img.shields.io/badge/Task-Semantic%20Segmentation-purple" alt="Task">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

**Gaia** (*Greek goddess of Earth*) - A production-ready deep learning pipeline for automated land cover classification from satellite imagery.

**Author:** Srikanth Akkaru  
Master's in Computer Science, University of South Florida  
📧 akkarusrikanth@gmail.com  
🔗 [GitHub](https://github.com/srikaanthh)

---

##  Project Overview

Gaia automatically segments satellite images into different land cover types using state-of-the-art deep learning. Built on U-Net architecture with EfficientNet-B0 encoder, it achieves high accuracy on large-scale satellite imagery through intelligent patch-based processing.

### Key Features

 **U-Net + EfficientNet-B0** - Powerful encoder-decoder architecture  
 **Patch-Based Processing** - Handles large satellite images efficiently  
 **Multi-Dataset Support** - LandCover.ai + optional DeepGlobe fusion  
 **Optimized Pipeline** - Fast training and inference with PyTorch  
 **Production Ready** - Complete logging, checkpointing, and evaluation  

---

## 🗺️ Land Cover Classes

The model segments images into the following categories:

| Class ID | Category | Description |
|----------|----------|-------------|
| 0 | Background | Unlabeled areas |
| 1 | Building | Urban structures |
| 2 | Woodland | Forests and trees |
| 3 | Water | Rivers, lakes, oceans |
| 4 | Road | Streets and highways |

---

## Architecture

```
Input Satellite Image (3 channels RGB)
           ↓
    EfficientNet-B0 Encoder
           ↓
       U-Net Decoder
           ↓
  Softmax Activation
           ↓
Output Segmentation Mask (N classes)
```

**Model:** U-Net with EfficientNet-B0 backbone  
**Framework:** PyTorch + Segmentation Models PyTorch (SMP)  
**Training Strategy:** Patch-based with configurable tile size (default 512×512)

---

## Repository Structure

```
gaia/
├── config/
│   └── config.yaml              # Main configuration file
├── data/                        # Datasets directory
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── test/
│   └── patches_512/            # Generated patches
├── models/                      # Saved model weights
├── output/                      # Predictions and visualizations
├── src/
│   ├── train.py                # Training script
│   ├── test.py                 # Inference script
│   └── utils/
│       ├── dataset.py          # Data loading utilities
│       ├── patching.py         # Image patchification
│       ├── preprocess.py       # Data augmentation
│       └── logger.py           # Logging utilities
├── notebooks/
│   └── training.ipynb          # Interactive training notebook
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Kaggle API credentials

### Installation

```bash
# Clone the repository
git clone https://github.com/srikaanthh/gaia.git
cd gaia

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Setup Kaggle Credentials

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Copy your kaggle.json
cp /path/to/kaggle.json ~/.kaggle/kaggle.json

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Download Datasets

```bash
# LandCover.ai dataset
kaggle datasets download -d adrianboguszewski/landcoverai -p data
unzip -q data/landcoverai.zip -d data

# Optional: DeepGlobe dataset
kaggle datasets download -d zfturbo/deepglobe-land-cover-classification-dataset -p data
unzip -q data/*.zip -d data/deepglobe
```

---

## Training

### Using Python Script

```bash
python src/train.py
```

The training pipeline will:
1. Patchify images into 512×512 tiles
2. Split data into train/validation (80/20)
3. Train U-Net model with EfficientNet-B0 encoder
4. Log metrics (IoU, Dice loss)
5. Save best model checkpoint

### Configuration

Edit `config/config.yaml` to customize:

```yaml
patch_size: 512          # Tile size for training
batch_size: 16           # Training batch size
init_lr: 0.0003         # Learning rate
epochs: 50              # Number of epochs
encoder: efficientnet-b0 # Encoder architecture
device: cuda            # cuda or cpu
```

---

## 🔍 Inference

### Run Predictions

```bash
python src/test.py
```

This will:
- Load the best trained model
- Process test images in batches
- Generate segmentation masks
- Save predictions and visualizations to `output/`

### Output

- **Predicted Masks:** `output/predicted_masks/`
- **Visualization Plots:** `output/prediction_plots/`

---

## Performance

### Metrics

- **IoU (Intersection over Union):** Measures overlap accuracy
- **Dice Coefficient:** Harmonic mean of precision and recall
- **Pixel Accuracy:** Overall classification accuracy


*Results vary based on dataset composition and training parameters*

---

## Datasets

### LandCover.ai
- High-resolution aerial imagery from Poland
- 41 images covering ~216 km²
- Native support with no preprocessing required

### DeepGlobe (Optional)
- Global land cover classification dataset
- Can be merged with LandCover.ai
- Requires label mapping (see notebook)

---

## Optimizations

- **Batched Patch Inference** - Processes multiple patches simultaneously
- **DataLoader Tuning** - Optimized `num_workers` and `pin_memory`
- **cuDNN Benchmark** - Automatic kernel optimization
- **Efficient Memory Usage** - Vectorized mask handling

See `OPTIMIZATION_SUMMARY.md` for detailed performance analysis.

---

## Troubleshooting

**CUDA Out of Memory?**
- Reduce `batch_size` in config
- Decrease `patch_size`

**Slow Data Loading?**
- Increase `num_workers` in train.py
- Use SSD for dataset storage

**Kaggle API Issues?**
- Verify `~/.kaggle/kaggle.json` exists
- Check file permissions: `chmod 600`

**Class Mismatch Errors?**
- Verify mask values with `np.unique(mask)`
- Update mapping in preprocessing

---

## Citation

If you use Gaia in your research, please cite:

```bibtex
@software{gaia2024,
  author = {Akkaru, Srikanth},
  title = {Gaia: Land Cover Semantic Segmentation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/srikaanthh/gaia}
}
```

**References:**
- Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch
- LandCover.ai: https://landcover.ai
- DeepGlobe: https://deepglobe.org


---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

**Note:** Datasets have their own licenses. Please review and comply with each dataset's terms.

---

## Acknowledgments

Inspired by "Segmenting the Earth: Challenges in Land Cover Classification" by Shirley Cheng (Stanford, 2023)

Built with:
- PyTorch
- Segmentation Models PyTorch (SMP)
- Albumentations
- NumPy & Matplotlib

---

**Made with 🌍 by Srikanth Akkaru**
