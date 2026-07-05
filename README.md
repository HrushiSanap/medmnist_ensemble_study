# Beyond Accuracy: Calibration and Uncertainty in Multi-Architecture Ensembles for MedMNIST Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20990733.svg)](https://doi.org/10.5281/zenodo.20990733)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-preprint%20under%20revision-orange)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Dataset](https://img.shields.io/badge/dataset-MedMNIST%20v2-1f6feb)
[![Model weights](https://img.shields.io/badge/weights-GitHub%20Releases-black?logo=github)](https://github.com/HrushiSanap/medmnist_ensemble_study/releases)

This repository provides the complete experimental codebase, pre-computed model predictions, numeric results, and generated figures for a study of heterogeneous deep-learning ensembles on the MedMNIST v2 collection. The work evaluates four architecturally diverse networks and two ensemble strategies across four medical imaging modalities at 224 x 224 resolution, and scores each configuration on the quality and structure of its predictive probabilities as well as on accuracy.

**Author:** Hrushikesh Sanap (independent researcher), Chhatrapati Sambhajinagar, India.

**Status:** preprint under revision. This is a major-revision resubmission to *Neural Computing and Applications* (NCAA, Springer). The numeric findings here reflect a full re-run of every experiment and supersede the original preprint. The preprint is archived on Zenodo at [10.5281/zenodo.20990733](https://doi.org/10.5281/zenodo.20990733).

> This is research code released for transparency and reproducibility. It uses public benchmark data and is not a clinical tool. None of the results should be read as a claim of clinical performance.

## Overview

The codebase implements an end-to-end pipeline for:

- **Multi-architecture training:** ConvNeXt-Base, Vision Transformer (ViT-Base), EfficientNetV2-M, and InceptionResNetV2, each fine-tuned from ImageNet weights.
- **Ensemble aggregation:** Soft Voting (parameter-free probability averaging) and Rigorous Stacking (a multi-layer perceptron meta-learner trained on validation predictions).
- **Three-layer evaluation:** discrimination (accuracy, AUC, macro precision, recall, F1, PR-AUC, log loss), predictive uncertainty (Shannon entropy of correct against incorrect predictions), and calibration (reliability diagrams and expected calibration error).
- **Interpretability:** Grad-CAM for the convolutional networks and last-layer attention for the Vision Transformer.

### Datasets

Four MedMNIST v2 datasets were chosen to span the clinical spectrum and a wide range of data regimes:

| Dataset | Modality | Task | Classes | Test size | Imbalance ratio |
| --- | --- | --- | --- | --- | --- |
| BloodMNIST | Microscopy | Multiclass | 8 | 3421 | 2.74 |
| BreastMNIST | Ultrasound | Binary | 2 | 156 | 2.71 |
| DermaMNIST | Dermatoscopy | Multiclass | 7 | 2005 | 58.66 |
| OrganAMNIST | Abdominal CT | Multiclass | 11 | 17778 | 4.54 |

### Key findings

- Every trained configuration exceeded the published MedMNIST v2 benchmark accuracy on all four datasets, with margins ranging from a fraction of a percentage point on the near-saturated BloodMNIST task to roughly fifteen percentage points on DermaMNIST.
- The ensembles matched or exceeded the strongest individual model in almost every case, the single exception being Rigorous Stacking on the small BreastMNIST set.
- **Accuracy and calibration are dissociated:** the most accurate configuration is rarely the best calibrated. Soft Voting attained the lower expected calibration error on all four datasets and the higher accuracy on three of them. The accuracy advantage of Rigorous Stacking was confined to the most imbalanced dataset, DermaMNIST, and came there at the cost of weaker probabilities.
- Predictive entropy placed incorrect predictions at higher uncertainty than correct ones throughout, the precondition for confidence-based triage, but the usable separation between the two distributions was held most consistently by Soft Voting. A strong aggregate calibration score could still mask a collapsed, non-informative per-prediction uncertainty.
- The reading on ensemble choice is that Soft Voting is the more dependable default. Rigorous Stacking helps only where discrete accuracy is the deployment priority and, on the data side, the base models are heterogeneous enough to offer structure worth learning while the validation set is large enough to fit the combiner. Validation-set size alone does not drive the result.

### Results at a glance

Headline numbers, taken from the re-run results. Benchmark accuracy is from the MedMNIST v2 evaluation. ECE is the expected calibration error from ten-bin quantile reliability diagrams (lower is better).

| Dataset | Benchmark accuracy | Best accuracy (configuration) | Lowest ECE (configuration) |
| --- | --- | --- | --- |
| BloodMNIST | 96.60% | 99.44% (Soft Voting) | 0.0035 (InceptionResNetV2); 0.0039 (Soft Voting) |
| BreastMNIST | 86.10% | 92.95% (Soft Voting) | 0.0389 (EfficientNetV2-M); 0.0455 (Soft Voting) |
| DermaMNIST | 76.80% | 91.67% (Rigorous Stacking) | 0.0163 (Soft Voting) |
| OrganAMNIST | 95.10% | 98.57% (Soft Voting) | 0.0073 (Soft Voting) |

The DermaMNIST accuracy gain is the largest in the study, yet it does not translate into reliable melanoma detection: the highest melanoma recall recorded is 0.7489, so roughly a quarter of melanomas are still assigned to benign nevi even by the strongest configuration. Headline accuracy and clinical usability are not the same thing, which is the central theme of the work.

## Repository structure

```
medmnist_ensemble_study/
│
├── notebooks/                 # Source code, organised per dataset (six notebooks each)
│   ├── bloodmnist/
│   ├── breastmnist/
│   ├── dermamnist/
│   └── organamnist/
│
├── predictions/               # Pre-computed softmax outputs (.npy), per dataset
│   ├── bloodmnist/
│   ├── breastmnist/
│   ├── dermamnist/
│   └── organamnist/
│
├── figures/                   # Per-dataset figures plus the final numbered set
│   ├── bloodmnist/
│   ├── breastmnist/
│   ├── dermamnist/
│   ├── organamnist/
│   └── publication_figures/   # fig1 to fig10 used in the manuscript
│
├── results/                   # Numeric outputs (CSV summaries and per-model JSON)
│   ├── bloodmnist/
│   ├── breastmnist/
│   ├── dermamnist/
│   └── organamnist/
│
├── requirements.txt
├── LICENSE                     # MIT License
└── README.md
```

Each top-level folder carries its own README describing naming conventions and contents in detail:

- `notebooks/README.md` documents the six-notebook per-dataset pipeline.
- `predictions/README.md` documents the `.npy` file naming and array shapes.
- `figures/README.md` documents the per-dataset figures and the figure-number mapping.
- `results/README.md` documents the CSV and JSON metric files.

## Experimental pipeline

The same pipeline is applied to all four datasets. Within each `notebooks/<dataset>/` folder:

1. `00_<dataset>_model_training` fine-tunes the four base architectures and exports validation and test probability vectors.
2. `01_<dataset>_ensemble_study_evaluation_analysis` builds both ensembles and computes the full three-layer evaluation, writing CSV and JSON to `results/<dataset>/` and panels to `figures/<dataset>/`.
3. `02_<dataset>_interpretability_ViT` and `03_<dataset>_interpretability_CNN` generate the attention and Grad-CAM maps.
4. `04_<dataset>_publication_figure` and `05_<dataset>_collage` assemble the manuscript composites and collages.

Key configuration: inputs at 224 x 224 x 3 (OrganAMNIST loaded at native 128 and bilinearly upsampled to 224 inside the pipeline); Adam optimiser with categorical cross-entropy; differentiated learning rates (2e-5 for ViT-Base, 1e-4 for the convolutional networks); up to 30 epochs with early stopping; mixed precision with the final dense layer in float32; no data augmentation and no class-imbalance correction, by design; a single global seed of 42 with operation determinism enabled. The stacking meta-learner is a small MLP (Dense 64, dropout 0.5, Dense 32, softmax) trained on the validation predictions.

## Reproducibility

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/HrushiSanap/medmnist_ensemble_study.git
cd medmnist_ensemble_study

# Install dependencies (Python 3.10)
pip install -r requirements.txt
```

### Fast verification (using the pre-computed predictions)

CPU is sufficient. The prediction arrays in `predictions/` let you regenerate every metric, reliability diagram, and entropy distribution without retraining.

1. Confirm the `.npy` arrays are present under `predictions/<dataset>/`.
2. Open `notebooks/<dataset>/01_<dataset>_ensemble_study_evaluation_analysis.ipynb`.
3. Run the evaluation sections. The notebook loads the saved predictions, fits the lightweight meta-learner in seconds, and rebuilds the metrics and plots.

### Full replication (training from scratch)

A GPU with 16 GB or more of VRAM is recommended. Base-model training takes several hours per dataset.

1. Run `00_<dataset>_model_training` to fine-tune the four base models and export their predictions.
2. Run the remaining notebooks (`01` through `05`) to evaluate, interpret, and assemble the figures.

## Trained model checkpoints

The full set of 16 trained checkpoints (four architectures across four datasets) is provided through the **[Releases](https://github.com/HrushiSanap/medmnist_ensemble_study/releases)** page. The weights are large (on the order of tens of gigabytes in total), so download only the file required for your analysis.

Files follow the convention `[Architecture]_[dataset].keras`, for example `ConvNeXtBase_bloodmnist.keras`.

```python
import tensorflow as tf

# Example: load the BloodMNIST ConvNeXt-Base model after downloading it from Releases
model = tf.keras.models.load_model('ConvNeXtBase_bloodmnist.keras')
model.summary()
```

A separate Releases tag will accompany this restructured branch once the updated checkpoints are uploaded.

## Citation

If you use this code or these findings, please cite the work. The citation will be updated on acceptance.

```bibtex
@misc{sanap2026beyondaccuracy,
  author       = {Sanap, Hrushikesh},
  title        = {Beyond Accuracy: Calibration and Uncertainty in Multi-Architecture
                  Ensembles for MedMNIST Classification},
  year         = {2026},
  howpublished = {Zenodo preprint and GitHub codebase},
  doi          = {10.5281/zenodo.18813204},
  url          = {https://github.com/HrushiSanap/medmnist_ensemble_study},
  note         = {Preprint under revision, Neural Computing and Applications}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds on the MedMNIST v2 benchmark. The original benchmark authors are acknowledged for providing standardised medical imaging data for machine-learning research, as are the authors of the source datasets from which MedMNIST v2 is derived.

## References

### Benchmark and datasets

- Yang, J., Shi, R., Wei, D., et al. (2023). MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41. https://doi.org/10.1038/s41597-022-01721-8
- Yang, J., Shi, R., Ni, B. (2021). MedMNIST classification decathlon: a lightweight AutoML benchmark for medical image analysis. *IEEE ISBI*, 191-195. https://doi.org/10.1109/ISBI48211.2021.9434062
- Doerrich, S., Di Salvo, F., Brockmann, J., Ledig, C. (2025). Rethinking model prototyping through the MedMNIST+ dataset collection. *Scientific Reports*, 15, 7669. https://doi.org/10.1038/s41598-025-92156-9
- Acevedo, A., Merino, A., Alferez, S., et al. (2020). A dataset for microscopic peripheral blood cell images. *Data in Brief*, 30, 105474. https://doi.org/10.1016/j.dib.2020.105474 (BloodMNIST)
- Al-Dhabyani, W., Gomaa, M., Khaled, H., Fahmy, A. (2020). Dataset of breast ultrasound images. *Data in Brief*, 28, 104863. https://doi.org/10.1016/j.dib.2019.104863 (BreastMNIST)
- Tschandl, P., Rosendahl, C., Kittler, H. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161. https://doi.org/10.1038/sdata.2018.161 (DermaMNIST)
- Codella, N. C., Gutman, D., Celebi, M. E., et al. (2018). Skin lesion analysis toward melanoma detection (ISIC). *IEEE ISBI*, 168-172. https://doi.org/10.1109/ISBI.2018.8363547 (DermaMNIST)
- Bilic, P., Christ, P. F., Li, H. B., et al. (2023). The liver tumor segmentation benchmark (LiTS). *Medical Image Analysis*, 84, 102680. https://doi.org/10.1016/j.media.2022.102680 (OrganAMNIST)

### Architectures

- Liu, Z., Mao, H., Wu, C. Y., et al. (2022). A ConvNet for the 2020s. *CVPR*, 11976-11986. https://doi.org/10.1109/CVPR52688.2022.01167
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: transformers for image recognition at scale. *ICLR*. https://doi.org/10.48550/arXiv.2010.11929
- Tan, M., Le, Q. (2021). EfficientNetV2: smaller models and faster training. *ICML*, 10096-10106. https://doi.org/10.48550/arXiv.2104.00298
- Szegedy, C., Ioffe, S., Vanhoucke, V., Alemi, A. (2017). Inception-v4, Inception-ResNet and the impact of residual connections on learning. *AAAI*, 31. https://doi.org/10.1609/aaai.v31i1.11231

### Ensembles, calibration, and interpretability

- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5, 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, 1321-1330. https://doi.org/10.48550/arXiv.1706.04599
- Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017). Grad-CAM: visual explanations from deep networks via gradient-based localization. *ICCV*, 618-626. https://doi.org/10.1109/ICCV.2017.74
- Chefer, H., Gur, S., Wolf, L. (2021). Transformer interpretability beyond attention visualization. *CVPR*, 782-791. https://doi.org/10.1109/CVPR46437.2021.00084
- Raghu, M., Zhang, C., Kleinberg, J., Bengio, S. (2019). Transfusion: understanding transfer learning for medical imaging. *NeurIPS*, 32. https://doi.org/10.48550/arXiv.1902.07208
