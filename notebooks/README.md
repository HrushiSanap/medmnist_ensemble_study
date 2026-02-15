# Notebooks & Source Code

This directory contains the complete experimental pipeline for the study **"Validation Starvation vs. Data Abundance."** The code is organized into five sequential notebooks, handling everything from initial data exploration to the training and evaluation of specific datasets.

## File Guide

| Notebook File | Description |
| :--- | :--- |
| **`00_eda_medmnist.ipynb`** | **Exploratory Data Analysis (EDA)**. <br>Generates class distribution plots, sample image grids, and pixel intensity histograms for all four datasets. Run this first to understand the data imbalances. |
| **`01_experiment_bloodmnist.ipynb`** | **BloodMNIST Experiments**. <br>Contains the full training loop for the 4 base models, the Meta-Learner (Stacking), and Soft Voting evaluation on the BloodMNIST dataset. |
| **`02_experiment_breastmnist.ipynb`** | **BreastMNIST Experiments**. <br>Focuses on the binary classification task under conditions of data scarcity. |
| **`03_experiment_dermamnist.ipynb`** | **DermaMNIST Experiments**. <br>Evaluates performance under extreme class imbalance ("Validation Starvation"). |
| **`04_experiment_organamnist.ipynb`** | **OrganAMNIST Experiments**. <br>Evaluates the models on a large-scale, relatively balanced dataset. |

## Usage
1.  **Dependencies:** Ensure you have installed the requirements listed in the root `requirements.txt`.
2.  **Order:** It is recommended to run `00_eda_medmnist.ipynb` first to download the data. The experiment notebooks (`01` through `04`) can be run independently of each other.
3.  **Outputs:** Running these notebooks will generate the model weights, predictions, and result plots.