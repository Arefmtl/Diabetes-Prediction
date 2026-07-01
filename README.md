# 🩺 Diabetes Prediction

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Machine learning project for diabetes prediction using medical data.**

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Dependencies](#-dependencies)
- [License](#-license)

## 🎯 Overview

This project predicts diabetes status using various machine learning algorithms based on medical data. It demonstrates a complete ML pipeline for healthcare applications.

### Key Highlights

- 📊 **Dataset**: Medical data with multiple health indicators
- 🤖 **Algorithms**: Comparison of classification models
- 🔧 **Feature Engineering**: Data preprocessing and transformation
- 📈 **Evaluation**: Accuracy, Precision, Recall, F1-Score

## ✨ Features

- **Multi-Model Comparison**: Tests various ML algorithms
- **Automated Pipeline**: End-to-end ML workflow
- **Feature Engineering**: Automatic data preprocessing
- **Cross-Validation**: Robust model evaluation
- **Visualization**: Results analysis and plotting

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

```bash
# Clone the repository
git clone https://github.com/Arefmtl/Diabetes-Prediction.git
cd Diabetes-Prediction

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Quick Start

```bash
# Run the main prediction script
python Diabet.py
```

### Example Code

```python
import pandas as pd
from Tool_box import DataProcessingTool, ClassificationTool, ModelEvaluationTool

# Load data
processor = DataProcessingTool()
data = processor.load_data("Dataset/diabetes.csv")

# Prepare data for ML
processed_data = processor.prepare_data_for_ml(
    data,
    target_column='diabetes',
    test_size=0.2,
    preprocessing_steps=['clean', 'encode', 'scale']
)

# Train models
classifier = ClassificationTool()
models = classifier.train_multiple_models(
    processed_data['X_train'],
    processed_data['y_train']
)

# Evaluate models
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_classification_models(
    models,
    processed_data['X_test'],
    processed_data['y_test']
)
```

## 📁 Project Structure

```
Diabetes-Prediction/
├── Diabet.py           # Main prediction script
├── Diabet.md           # Detailed documentation
├── Dataset/
│   └── diabetes.csv    # Medical data
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 📊 Results

The project compares multiple classification algorithms:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| Gradient Boosting | - | - | - | - |
| XGBoost | - | - | - | - |

*Run the script to see actual results*

## 🛠️ Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Plotting and visualization

## 🔧 ML Tools

This project uses tools from [TOOL-BOX](https://github.com/Arefmtl/TOOL-BOX) — a comprehensive ML toolbox with 8 specialized modular tools.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Ali Kazemi** - [@Arefmtl](https://github.com/Arefmtl)

Project Link: [https://github.com/Arefmtl/Diabetes-Prediction](https://github.com/Arefmtl/Diabetes-Prediction)
