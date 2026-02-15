# MedMNIST+ Ensemble Study

**Official implementation of "Validation Starvation vs. Data Abundance"**

This repository provides the complete experimental codebase, trained model predictions, and generated visualizations for benchmarking heterogeneous ensemble strategies on the MedMNIST v2 collection. The study evaluates state-of-the-art deep learning architectures and proposes decision heuristics for selecting between parameter-free ensemble methods and meta-learning approaches based on validation set availability.

## Overview

The codebase implements an end-to-end pipeline for:

- **Multi-Architecture Training**: ConvNeXt-Base, Vision Transformer (ViT-Base), EfficientNetV2-M, and InceptionResNetV2
- **Ensemble Aggregation**: Soft Voting (parameter-free) and Rigorous Stacking (meta-learning)
- **Model Safety Evaluation**: Calibration analysis via reliability diagrams and uncertainty quantification through entropy distributions

### Key Contributions

- Achieves state-of-the-art performance across four MedMNIST datasets
- Demonstrates +15% improvement on DermaMNIST classification
- Provides calibrated uncertainty estimates for clinical AI applications
- Establishes decision criteria for ensemble method selection based on validation set size

## Repository Structure

```
medmnist_ensemble_study/
│
├── notebooks/                          # Source code and experimental logs
│   ├── 00_eda_medmnist.ipynb           # Data exploration and visualization
│   ├── 01_experiment_bloodmnist.ipynb  # BloodMNIST training pipeline
│   ├── 02_experiment_breastmnist.ipynb # BreastMNIST training pipeline
│   ├── 03_experiment_dermamnist.ipynb  # DermaMNIST training pipeline
│   └── 04_experiment_organamnist.ipynb # OrganAMNIST training pipeline
│
├── predictions/                        # Pre-computed model outputs (.npy)
│   ├── bloodmnist/                     # BloodMNIST predictions
│   │   ├── ConvNeXtBase_preds.npy      # Test set probabilities
│   │   ├── ConvNeXtBase_val_preds.npy  # Validation set probabilities
│   │   └── ... (8 files per dataset)
│   ├── breastmnist/
│   ├── dermamnist/
│   └── organamnist/
│
├── plots/                              # Generated visualizations (.png)
│   ├── 00_eda/                         # Exploratory data analysis
│   ├── 01_bloodmnist/                  # BloodMNIST results
│   ├── 02_breastmnist/                 # BreastMNIST results
│   ├── 03_dermamnist/                  # DermaMNIST results
│   └── 04_organamnist/                 # OrganAMNIST results
│
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
└── README.md                           # Documentation
```

## Experimental Pipeline

### Data Exploration (Notebook 00)

The exploratory analysis notebook performs:

- Automated download of MedMNIST v2 datasets in `.npz` format
- Class label decoding and statistical analysis
- Sample visualization through random montage generation
- Class distribution analysis to quantify imbalance ratios

Output artifacts are saved to `plots/00_eda/`.

### Model Training and Evaluation (Notebooks 01-04)

Each experiment notebook executes a complete machine learning workflow for its respective dataset:

1. **Data Preprocessing**: Load and transform data (resize to 224×224, normalize pixel values)
2. **Model Training**: Train four heterogeneous architectures with ImageNet pre-training
3. **Prediction Generation**: Save probability distributions for both validation and test sets
4. **Ensemble Construction**: Implement soft voting and train stacking meta-learner
5. **Performance Visualization**: Generate confusion matrices, calibration plots, and uncertainty distributions

## Prediction File Reference

Pre-computed predictions enable result verification without GPU requirements. All predictions are stored as NumPy arrays in `.npy` format.

### File Organization

Each dataset directory contains **8 files** corresponding to 4 base models × 2 data splits:

| File Suffix | Data Split | Purpose |
|-------------|-----------|---------|
| `_val_preds.npy` | Validation Set | Training data for stacking meta-learner (prevents data leakage) |
| `_preds.npy` | Test Set | Final evaluation metrics and visualization generation |

### Usage Example

```python
import numpy as np

# Load test predictions for ConvNeXt on BloodMNIST
preds = np.load('predictions/bloodmnist/ConvNeXtBase_preds.npy')
print(f"Prediction shape: {preds.shape}")  # Expected: (3421, 8)
```

## Visualization Reference

Generated plots are organized by experimental stage. Each dataset folder contains performance metrics visualized through confusion matrices, reliability diagrams, and entropy distributions.

### Model Identification System

Visualization files use numeric identifiers in their filenames:

| ID | Model/Strategy | Type | Description |
|----|---------------|------|-------------|
| 0 | Soft Voting | Ensemble | Uniform average of all base model predictions |
| 1 | ConvNeXt-Base | Base Model | Modern CNN with transformer-inspired design principles |
| 2 | ViT-Base | Base Model | Vision Transformer (ImageNet-21k pre-trained) |
| 3 | EfficientNetV2-M | Base Model | Efficiency-optimized convolutional network |
| 4 | InceptionResNetV2 | Base Model | Multi-scale architecture with residual connections |
| 5 | Rigorous Stacking | Ensemble | Logistic regression meta-learner trained on validation predictions |

**Example**: The file `plots/03_dermamnist/derma5.png` contains visualizations for the stacking ensemble (ID=5) on the DermaMNIST dataset.

## Reproducibility Guide

### Prerequisites

```bash
# Clone repository
git clone https://github.com/HrushiSanap/medmnist_ensemble_study.git
cd medmnist_ensemble_study

# Install dependencies
pip install -r requirements.txt
```

### Full Replication (Training from Scratch)

**Hardware Requirements**: GPU with 16GB+ VRAM recommended

1. Execute `00_eda_medmnist.ipynb` to download datasets and generate exploratory visualizations
2. Run experiment notebooks (01-04) sequentially or in parallel
3. Each notebook will:
   - Train models from scratch (several hours per dataset)
   - Save predictions to `predictions/` directory
   - Generate performance plots in `plots/` subdirectories

### Fast Verification (Using Pre-computed Predictions)

**Hardware Requirements**: CPU-only execution supported

1. Ensure `predictions/` directory contains provided `.npy` files
2. Open any experiment notebook (01-04)
3. Skip or comment out all `model.fit()` cells for base models
4. Execute the ensemble evaluation sections
5. The notebook will:
   - Load existing predictions from disk
   - Train lightweight meta-learner (seconds on CPU)
   - Regenerate all plots and metrics

This approach allows complete verification of results without computational overhead of model training.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{medmnist_ensemble_study,
  author = {Sanap, Hrushi},
  title = {Validation Starvation vs. Data Abundance: Ensemble Strategies for Medical Image Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HrushiSanap/medmnist_ensemble_study}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds upon the MedMNIST v2 benchmark dataset. We acknowledge the original authors for providing standardized medical imaging data for machine learning research.

## References

### Dataset

If you use this codebase, please cite the official MedMNIST v2 paper:

> Yang, J., Shi, R., Wei, D. et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." *Scientific Data* 10, 41 (2023).

```bibtex
@article{medmnistv2,
    title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={Scientific Data},
    volume={10},
    number={1},
    pages={41},
    year={2023},
    publisher={Nature Publishing Group}
}

```

### Model Architectures

This study utilizes the following architectures:

* **ConvNeXt:** Liu, Z. et al. "A ConvNet for the 2020s." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11976–11986 (2022).
* **Vision Transformer (ViT):** Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *International Conference on Learning Representations (ICLR)* (2021).
* **EfficientNetV2:** Tan, M. & Le, Q. "EfficientNetV2: Smaller Models and Faster Training." *International Conference on Machine Learning (ICML)* (2021).
* **InceptionResNetV2:** Szegedy, C. et al. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning." *Proceedings of the AAAI Conference on Artificial Intelligence*, 31(1) (2017).