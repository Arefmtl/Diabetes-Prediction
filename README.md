# 🩺 Diabetes Prediction — Full ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/Arefmtl/Diabetes-Prediction/pulls)

A **standalone, production-ready** machine learning pipeline for binary classification of diabetes using the Pima Indians Diabetes Dataset. Designed to demonstrate rigorous methodology for technical hiring assessments.

---

## 📊 Pipeline Overview

```
┌──────────┐   ┌────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
│  Load    │ → │  EDA   │ → │ SMOTE +  │ → │ 9 + 6    │ → │  Metrics  │
│  Data    │   │ 6 plots│   │  Scaling │   │ Classifiers│   │ + Viz     │
└──────────┘   └────────┘   └──────────┘   └──────────┘   └───────────┘
```

## ✨ Features

| Feature | Detail |
|---------|--------|
| **Exploratory Data Analysis** | 6 publication‑quality visualisations (distributions, correlations, box‑plots, pair‑plot, missing‑data) |
| **Class Imbalance Handling** | SMOTE (Synthetic Minority Oversampling) |
| **15 Classifiers** | 9 individual (LR, KNN, DT, RF, SVM, NB, GBDT, ET, MLP) + 6 ensemble (AdaBoost, Bagging, Soft‑Voting, Stacking ×2) |
| **Stratified Cross‑Validation** | 5‑fold CV with F1‑score & accuracy |
| **Evaluation Metrics** | Accuracy, Precision, Recall, **F1‑Score** (primary) |
| **Confusion Matrices** | Grid of all 15 models |
| **PCA Visualisation** | 2‑D projection with explained variance |
| **Feature Importance** | Bar‑plot for tree‑based models |
| **Comprehensive Logging** | Every step logged with timestamps |
| **CLI Arguments** | `--no-eda` · `--no-pca` · `--data` · `--out` |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Arefmtl/Diabetes-Prediction.git
cd Diabetes-Prediction

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python diabetes_prediction.py

# Run without EDA / PCA (faster)
python diabetes_prediction.py --no-eda --no-pca
```

### Output Structure

```
output/
├── figures/
│   ├── 01_outcome_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_boxplots_vs_outcome.png
│   ├── 05_pairplot.png
│   ├── 06_zero_values.png
│   ├── 07_confusion_matrices.png
│   ├── 08_pca_projection.png
│   └── 09_feature_importance.png
└── results.csv
```

## 📈 Key Findings

- **Best model**: *varies by run — typically Gradient Boosting or Random Forest* with **F1‑score > 0.75**
- **Top predictive features**: Glucose, BMI, Age, DiabetesPedigreeFunction
- SMOTE consistently improves recall for the minority (diabetic) class
- Stratified CV ensures robust evaluation despite class imbalance (~35% diabetic)

## 📁 Repository Structure

```
├── diabetes_prediction.py   ← Main pipeline (standalone, no external deps)
├── requirements.txt         ← Python dependencies
├── Dataset/
│   └── diabetes.csv         ← Pima Indians Diabetes Dataset
├── output/                  ← Generated figures & results
│   ├── figures/
│   └── results.csv
└── README.md
```

## 📖 Dataset

**Pima Indians Diabetes Database** (National Institute of Diabetes and Digestive and Kidney Diseases)

- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples**: 768
- **Features**: 8 (Glucose, BMI, Age, Insulin, BloodPressure, SkinThickness, DiabetesPedigreeFunction, Pregnancies)
- **Target**: Binary (0 = No Diabetes, 1 = Diabetes)

## 📄 License

MIT — Free to use, modify, and distribute.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/Arefmtl">Ali Kazemi</a>
</p>
