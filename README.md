# Sequential Feature Selection (Custom Implementation)

##  Overview

This project provides a **custom implementation of Sequential Feature Selection (SFS)** in Python.

It is designed as a modular and reusable mini-library that mimics feature selection techniques used in machine learning workflows.

The implementation supports:

* Forward Selection
* Backward Elimination
* Multiple evaluation metrics

---

##  What is Sequential Feature Selection?

Sequential Feature Selection (SFS) is a greedy algorithm used to select the most relevant subset of features by:

* Adding features one by one (**forward**)
* Removing features one by one (**backward**)

At each step, the model evaluates different feature subsets and selects the best based on a scoring metric.

---

##  Project Structure

```id="p3r0zw"
Subset_Forward_Selection/
│
├── Selector.py        # Main feature selection logic
├── metrics.py         # Evaluation metrics (Accuracy, Recall, Precision, R2)
├── utils.py           # Helper functions (e.g., cloning estimator)
├── __init__.py        # Makes the folder behave like a Python package
```

---

##  Features

* Modular design (separated into logic, metrics, and utilities)
* Works with any **scikit-learn compatible model**
* Supports multiple metrics:

  * Accuracy
  * Recall
  * Precision
  * R2 Score
* Easy to extend and modify

---

##  Installation

Make sure you have the required libraries:

```bash id="q3n7gk"
pip install numpy scikit-learn
```

---

##  Usage Example

```python id="1r1m8p"
from Subset_Forward_Selection.Selector import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Define model
model = LogisticRegression(max_iter=10000)

# Initialize selector
sfs = SequentialFeatureSelector(
    estimator=model,
    n_features_to_select=5,
    direction="forward",
    scoring="Accuracy"
)

# Fit model
sfs.fit(X, y)

# Results
print("Selected Features:", sfs.selected_features)
print("Support Mask:", sfs.get_support())

# Transform dataset
X_new = sfs.transform(X)
print("New Shape:", X_new.shape)
```

---

##  Modules Description

###  Selector.py

Contains the core implementation of the `SequentialFeatureSelector` class:

* Forward selection
* Backward elimination
* Feature evaluation logic

---

###  metrics.py

Handles evaluation metrics:

* Accuracy
* Recall
* Precision
* R2 Score

---

###  utils.py

Contains helper utilities:

* `clone_estimator`: safely clones sklearn models

---

##  Use Cases

* Feature engineering experiments
* Educational purposes (understanding SFS)
* Custom ML pipelines

---

##  Notes

* This implementation evaluates performance on the **same training data** (no cross-validation).
* Designed for learning and experimentation, not production.

---

##  Author

**Hussen Sabry**


