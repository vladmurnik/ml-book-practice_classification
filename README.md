# MNIST Study Project

This repository contains a practical implementation based on a machine learning book chapter, focused on solving the MNIST handwritten digit classification problem.

The goal of the project is to understand core ML concepts through hands-on experimentation: data preprocessing, model training, evaluation, and comparison of different approaches.

---

## Project Structure

```

.
├── .venv/
├── dataset/
├── img/
│   ├── cleaned_digit_example_plot.png
│   ├── confusion_matrix_errors_plot.png
│   ├── noisy_digit_example_plot.png
│   ├── precision_recall_vs_threshold_plot.png
│   ├── roc_curve_comparison_plot.png
│   ├── roc_curve_plot.png
│   └── some_digit_plot.png
├── models/
│   └── knn_model.pkl
├── .gitignore
├── main.py
└── MNIST.ipynb

````

---

## Features

- Loading and preprocessing the MNIST dataset  
- Data visualization (sample digits, noisy and cleaned images)  
- Binary classification (digit 5 vs others)  
- Multiclass classification  
- Multilabel classification  
- Model evaluation using:
  - Precision / Recall  
  - F1-score  
  - Confusion matrix  
  - ROC curve  
  - Precision-Recall curve  
- Model comparison:
  - SGD Classifier  
  - Random Forest  
  - KNN  
  - SVM  
- Noise removal from images  

---

## Saved Artifacts

- `img/` — generated plots and visualizations  
- `models/knn_model.pkl` — trained KNN model  

---

## Requirements

- Python 3.9+  
- Jupyter Notebook or JupyterLab  

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn jupyter
````

---

## Dataset

Place the MNIST dataset inside:

```
dataset/
```

Make sure paths in `main.py` or the notebook match your local setup.

---

## Usage

Run notebook:

```bash
jupyter notebook MNIST.ipynb
```

Run script:

```bash
python main.py
```

---

## Purpose

This repository is intended for learning and experimentation. It follows a classic MNIST workflow to help understand:

* model behavior
* evaluation metrics
* error analysis
* working with image data in scikit-learn

---

## Notes

* This is a study project, not a production system
* Results may vary depending on environment