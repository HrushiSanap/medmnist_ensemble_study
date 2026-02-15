# Result Plots & Visualizations

This directory contains the generated performance visualizations, including **Confusion Matrices**, **Reliability Diagrams** (Calibration Curves), and **Uncertainty Distributions** (Entropy). The folder named `00_eda` contains the plots generated while doing exploratory data analysis to study class distribuitions. the plots for individual datasets are stored in folder named `01_bloodmnist`, `02_breastmnist`, `03_dermamnist` and `04_organamnist`

## Naming Convention
The image files follow the format: `[dataset][ID].png`. 

* **Prefix:** `blood`, `breast`, `derma`, or `organ` (indicating the dataset).
* **Suffix (Number):** Indicates the specific model or ensemble strategy used.

### Model ID Key

| ID Number | Model / Strategy | Type |
| :--- | :--- | :--- |
| **0** | **Soft Voting Ensemble** | Parameter-Free Aggregation |
| **1** | **ConvNeXt-Base** | Base Model (Modernized CNN) |
| **2** | **ViT-Base** | Base Model (Vision Transformer) |
| **3** | **EfficientNetV2-M** | Base Model (CNN) |
| **4** | **InceptionResNetV2** | Base Model (Hybrid CNN) |
| **5** | **Rigorous Stacking** | Meta-Learning Ensemble |

### Examples
* **`blood0.png`**: Performance plots for the **Soft Voting** ensemble on BloodMNIST.
* **`derma5.png`**: Performance plots for the **Stacking** ensemble on DermaMNIST.
* **`organ1.png`**: Performance plots for the standalone **ConvNeXt-Base** model on OrganAMNIST.