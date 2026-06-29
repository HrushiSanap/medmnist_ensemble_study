# Figures and Visualisations

This directory holds every generated figure for the study, in two forms: per-dataset working files, and the final numbered set used in the manuscript. Most figures are provided as both a raster PNG and a true-vector SVG, so they can be inspected quickly or re-exported at any resolution.

## Directory layout

```
figures/
├── bloodmnist/
├── breastmnist/
├── dermamnist/
├── organamnist/
└── publication_figures/
```

## Per-dataset folders

Each of the four dataset folders contains the figures for that modality. The block below lists the BloodMNIST contents; the other three folders mirror it, with the dataset prefix changed and the composite figure renumbered (see the mapping table further down).

| File | Contents |
| --- | --- |
| `ConvNeXtBase_evaluation.png` | Three-panel evaluation for ConvNeXt-Base: confusion matrix, Shannon-entropy distribution (correct against incorrect), and reliability diagram. |
| `ViT-Base_evaluation.png` | Same three panels for ViT-Base. |
| `EfficientNetV2M_evaluation.png` | Same three panels for EfficientNetV2-M. |
| `InceptionResNetV2_evaluation.png` | Same three panels for InceptionResNetV2. |
| `SoftVoting_evaluation.png` | Same three panels for the Soft Voting ensemble. |
| `RigorousStacking_evaluation.png` | Same three panels for the Rigorous Stacking ensemble. |
| `ConvNeXtBase_gradcam.png` / `.svg` | Grad-CAM panels for ConvNeXt-Base (five correct and five incorrect cases). |
| `EfficientNetV2M_gradcam.png` / `.svg` | Grad-CAM panels for EfficientNetV2-M. |
| `InceptionResNetV2_gradcam.png` / `.svg` | Grad-CAM panels for InceptionResNetV2. |
| `ViT-Base_attention.png` / `.svg` | Last-layer CLS attention panels for ViT-Base. |
| `<dataset>_interpretability_collage.png` / `.svg` | The four interpretability blocks stacked into one combined figure. |
| `Fig<N>.png` / `.svg` | The six-row, three-column evaluation composite for the dataset (the manuscript figure). |
| `Fig<N>_900.png` | A 900-dpi raster of that composite, for print-quality use. |

The six `*_evaluation.png` panels are the building blocks; the composite `Fig<N>` is what those panels assemble into.

## Final numbered figures (`publication_figures/`)

This folder collects the final manuscript figures as `fig1` through `fig10`, in PNG and (where applicable) SVG. Two of them are global rather than per-dataset.

| File | Figure | Description |
| --- | --- | --- |
| `fig1.png` | Fig. 1 | Representative image samples from the four datasets (a 5 x 5 grid per modality). |
| `fig2.png` / `fig2.svg` | Fig. 2 | End-to-end methodology diagram. |
| `Fig3.png` / `Fig3.svg` | Fig. 3 | BloodMNIST evaluation composite. |
| `fig4.png` / `fig4.svg` | Fig. 4 | BloodMNIST interpretability collage. |
| `Fig5.png` / `Fig5.svg` | Fig. 5 | BreastMNIST evaluation composite. |
| `fig6.png` / `fig6.svg` | Fig. 6 | BreastMNIST interpretability collage. |
| `Fig7.png` / `Fig7.svg` | Fig. 7 | DermaMNIST evaluation composite. |
| `fig8.png` / `fig8.svg` | Fig. 8 | DermaMNIST interpretability collage. |
| `Fig9.png` / `Fig9.svg` | Fig. 9 | OrganAMNIST evaluation composite. |
| `fig10.png` / `fig10.svg` | Fig. 10 | OrganAMNIST interpretability collage. |

## Per-dataset figure-number mapping

Each dataset contributes one evaluation composite and one interpretability collage to the manuscript. The mapping between the per-dataset working files and the numbered publication figures is:

| Dataset | Evaluation composite | Interpretability collage |
| --- | --- | --- |
| BloodMNIST | `Fig3` (Fig. 3) | `fig4` (Fig. 4) |
| BreastMNIST | `Fig5` (Fig. 5) | `fig6` (Fig. 6) |
| DermaMNIST | `Fig7` (Fig. 7) | `fig8` (Fig. 8) |
| OrganAMNIST | `Fig9` (Fig. 9) | `fig10` (Fig. 10) |

So within `figures/dermamnist/` the composite is named `Fig7`, and its interpretability collage is `dermamnist_interpretability_collage`, which appears in `publication_figures/` as `fig8`.

## How to read each panel

- **Confusion matrix:** class-wise prediction error rates. Strongly diagonal matrices indicate few errors; off-diagonal cells reveal which classes are confused.
- **Entropy distribution:** Shannon entropy of each prediction, plotted separately for correct (blue) and incorrect (orange) predictions. A clear separation, with incorrect predictions sitting at higher entropy, is the precondition for confidence-based triage.
- **Reliability diagram:** predicted confidence against observed accuracy, using ten quantile bins. The dashed diagonal is perfect calibration; points above it indicate underconfidence and points below it indicate overconfidence.
- **Grad-CAM and attention maps:** warmer colours mark higher attribution weight. Each panel is titled with its correctness tag, true label, and predicted label with confidence.

## Notes

- SVG files are true vectors exported from matplotlib and are suitable for production typesetting. The PNG files are large rasters and exist for quick viewing.
- File sizes vary widely because the interpretability collages embed many overlay images.

Figures are released for transparency. They illustrate model behaviour on public benchmark data and are not clinical guidance.
