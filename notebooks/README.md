# Notebooks and Source Code

This directory holds the complete experimental pipeline for the study *Beyond Accuracy: Calibration and Uncertainty in Multi-Architecture Ensembles for MedMNIST Classification*. The code is now organised by dataset rather than by experiment stage. Each of the four datasets has its own subfolder containing the same six sequential notebooks, so that training, evaluation, interpretability, and figure assembly for one modality live together.

## Directory layout

```
notebooks/
├── bloodmnist/
├── breastmnist/
├── dermamnist/
└── organamnist/
```

Every subfolder contains the same six notebooks, prefixed with the dataset name. The block below uses BloodMNIST as the example; the other three folders mirror it with their own prefixes (for example `02_breastmnist_interpretability_ViT.ipynb`).

## Notebook guide

| Notebook | Role |
| --- | --- |
| `00_<dataset>_model_training.ipynb` | Trains the four base architectures (ConvNeXt-Base, ViT-Base, EfficientNetV2-M, InceptionResNetV2) from ImageNet weights, then exports softmax probability vectors for both the validation and test splits. The validation predictions later train the stacking meta-learner; the test predictions feed every downstream metric. |
| `01_<dataset>_ensemble_study_evaluation_analysis.ipynb` | Builds both ensembles (Soft Voting and Rigorous Stacking), then computes the full evaluation suite for all six configurations: discrimination metrics, Shannon-entropy uncertainty, and reliability and expected calibration error. Writes the numeric outputs to `results/<dataset>/` and the per-model evaluation panels to `figures/<dataset>/`. |
| `02_<dataset>_interpretability_ViT.ipynb` | Produces last-layer classification-token (CLS) attention maps for ViT-Base, for five correctly classified and five misclassified test images. |
| `03_<dataset>_interpretability_CNN.ipynb` | Produces Grad-CAM maps for the three convolutional networks, using each network's final convolutional layer, on the same correct-against-incorrect sampling protocol. |
| `04_<dataset>_publication_figure.ipynb` | Assembles the six-row, three-column evaluation composite (confusion matrix, entropy distribution, reliability diagram for each configuration) into the manuscript figure for that dataset. |
| `05_<dataset>_collage.ipynb` | Stacks the per-model Grad-CAM and attention panels into the single interpretability collage for that dataset, using `svgutils` for true-vector output. |

## Recommended order

Within a dataset folder the notebooks are numbered in dependency order:

1. Run `00_*_model_training` first if reproducing from scratch. This step requires a GPU and several hours per dataset. If you only want to verify the published numbers, skip it and use the pre-computed arrays in `predictions/<dataset>/` instead (see that folder's README).
2. Run `01_*_ensemble_study_evaluation_analysis` to regenerate every metric and per-model panel. This step is CPU-friendly when the prediction arrays are already present, since the meta-learner is a small multi-layer perceptron.
3. Run `02_*` and `03_*` to regenerate the interpretability maps.
4. Run `04_*` and `05_*` to rebuild the manuscript composites and collages.

The four dataset folders are independent of one another and can be run in any order.

## Notes on reproducibility

- A single global seed of 42 (Python, NumPy, TensorFlow) is set, with TensorFlow operation determinism enabled. Results come from a single training run per model under this seed.
- Base-model training used the legacy Keras 2 backend under TensorFlow 2.19.0; local evaluation and interpretability ran under TensorFlow 2.21.0 with OpenCV 4.13.0. See the root `requirements.txt`.
- All images are processed at 224 x 224 x 3. OrganAMNIST is loaded at its native 128 resolution and bilinearly upsampled to 224 inside the data pipeline, and the interpretability notebooks reproduce that same upsampling so the explanation reflects the exact input the model received.
- No data augmentation and no class-imbalance correction are applied anywhere, by design, so that differences are attributable to architecture and ensemble strategy rather than to data-side interventions.

This is research code released for transparency and reproducibility. It is not a clinical tool and is not intended for diagnostic use.
