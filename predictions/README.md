# Pre-computed Predictions

To ensure reproducibility without requiring users to retrain the heavy base models (approx. 24GB of weights), we provide the raw prediction outputs in `.npy` (NumPy) format. These files will allow to reproduce the ensemble results, calibration plots, and metric calculations instantly.

## Directory Structure

The predictions are organized into four subdirectories corresponding to the datasets:

```text
predictions/
├── bloodmnist/   # BloodMNIST predictions
├── breastmnist/  # BreastMNIST predictions
├── dermamnist/   # DermaMNIST predictions
└── organamnist/   # OrganAMNIST predictions