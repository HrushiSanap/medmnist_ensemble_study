# Pre-computed Predictions

To allow the results, calibration analysis, and ensemble metrics to be reproduced without retraining the heavy base models, this directory provides the raw softmax outputs as NumPy `.npy` arrays. Loading these files lets you regenerate every number in the manuscript on a CPU in seconds.

## Directory layout

```
predictions/
├── bloodmnist/    # 8 classes
├── breastmnist/   # 2 classes
├── dermamnist/    # 7 classes
└── organamnist/   # 11 classes
```

## File naming

Each dataset folder contains eleven files: two splits for each of the four base models, one test-split file for each of the two ensembles, and the ground-truth label vector.

| File | Contents | Shape (BloodMNIST example) |
| --- | --- | --- |
| `ConvNeXtBase_val_preds.npy` | ConvNeXt-Base validation probabilities | `(1712, 8)` |
| `ConvNeXtBase_test_preds.npy` | ConvNeXt-Base test probabilities | `(3421, 8)` |
| `ViT-Base_val_preds.npy` | ViT-Base validation probabilities | `(1712, 8)` |
| `ViT-Base_test_preds.npy` | ViT-Base test probabilities | `(3421, 8)` |
| `EfficientNetV2M_val_preds.npy` | EfficientNetV2-M validation probabilities | `(1712, 8)` |
| `EfficientNetV2M_test_preds.npy` | EfficientNetV2-M test probabilities | `(3421, 8)` |
| `InceptionResNetV2_val_preds.npy` | InceptionResNetV2 validation probabilities | `(1712, 8)` |
| `InceptionResNetV2_test_preds.npy` | InceptionResNetV2 test probabilities | `(3421, 8)` |
| `SoftVoting_test_preds.npy` | Soft Voting ensemble test probabilities | `(3421, 8)` |
| `RigorousStacking_test_preds.npy` | Rigorous Stacking ensemble test probabilities | `(3421, 8)` |
| `y_true_flat.npy` | Integer ground-truth labels for the test split | `(3421,)` |

### Conventions

- A `_val_preds` file holds probabilities on the **validation** split. These are the inputs the stacking meta-learner is trained on, kept strictly separate from the test split to prevent leakage.
- A `_test_preds` file holds probabilities on the **test** split. These drive every final metric, reliability diagram, and entropy distribution.
- The two ensembles have **test files only**. Soft Voting averages the four base-model test vectors; Rigorous Stacking is trained on the four base-model validation vectors and then evaluated on the test split.
- `y_true_flat.npy` is a one-dimensional array of integer class indices that match the class ordering documented in `results/<dataset>/class_distribution.csv` and in the manuscript label tables.

## Per-dataset shapes

The class count and split sizes differ by dataset, so the array shapes differ accordingly.

| Dataset | Classes | Validation rows | Test rows |
| --- | --- | --- | --- |
| BloodMNIST | 8 | 1712 | 3421 |
| BreastMNIST | 2 | 78 | 156 |
| DermaMNIST | 7 | 1003 | 2005 |
| OrganAMNIST | 11 | 6491 | 17778 |

So, for example, `predictions/organamnist/ViT-Base_test_preds.npy` has shape `(17778, 11)` and `predictions/breastmnist/SoftVoting_test_preds.npy` has shape `(156, 2)`.

## Usage example

```python
import numpy as np

# Load ConvNeXt-Base test probabilities for BloodMNIST
preds = np.load('predictions/bloodmnist/ConvNeXtBase_test_preds.npy')
print(preds.shape)        # (3421, 8)

# Ground-truth labels
y_true = np.load('predictions/bloodmnist/y_true_flat.npy')
print(y_true.shape)       # (3421,)

# Reconstruct Soft Voting directly from the four base models
models = ['ConvNeXtBase', 'ViT-Base', 'EfficientNetV2M', 'InceptionResNetV2']
stack = np.stack([
    np.load(f'predictions/bloodmnist/{m}_test_preds.npy') for m in models
])
soft_vote = stack.mean(axis=0)
acc = (soft_vote.argmax(1) == y_true).mean()
print(f'Soft Voting accuracy: {acc:.4f}')
```

This recomputed Soft Voting array should match the supplied `SoftVoting_test_preds.npy` to numerical precision.

Predictions are released for verification only. They are derived from public benchmark data and are not a clinical tool.
