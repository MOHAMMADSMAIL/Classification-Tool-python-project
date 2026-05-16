[README.md](https://github.com/user-attachments/files/27858381/README.md)
 #ML Classification Tool

> **Built to understand, not just to use.**

Most ML courses teach students to call `sklearn.fit()` without explaining what happens inside. This notebook solves that — it implements KNN, Naive Bayes, and Decision Tree **from scratch**, so every step of the learning process is visible and traceable.

---

## The Problem

When students use scikit-learn, they treat algorithms as black boxes:

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)   # What actually happens here?
```

This works, but it builds no real understanding. A student who only uses sklearn cannot explain *why* KNN fails on high-dimensional data, or *what* Naive Bayes assumes about feature distributions.

## The Solution

This notebook replaces the black box with transparent code. Every algorithm is written using only Python built-ins and NumPy — no sklearn for training. A student running this notebook will see exactly how:

- KNN measures distance and picks neighbors
- Naive Bayes computes probabilities per class
- A Decision Tree chooses split points using Gini impurity

The same pipeline works on **any CSV dataset**, making it reusable across different courses and experiments.

---

## Full Pipeline

| Step | What Happens |
|---|---|
| Load | Reads CSV from local path or Google Drive link |
| Clean | Removes duplicates, fills missing values (mean / mode) |
| Encode | Maps text columns to integers using dictionaries |
| Split & Scale | 80/20 split, standardization using train-set stats only |
| Train | Three classifiers built manually with NumPy |
| Evaluate | Accuracy, confusion matrix, side-by-side comparison chart |

---

## Algorithms — Implemented From Scratch

**K-Nearest Neighbors**
Computes Euclidean distance between the test row and every training row, then predicts by majority vote among the *k* closest neighbors. Uses `(distance, label)` tuples — no external sorting library.

**Gaussian Naive Bayes**
Estimates the mean and standard deviation of each feature per class during training. At prediction time, picks the class with the highest log-posterior probability using the Gaussian probability density formula.

**Decision Tree**
Builds a recursive binary tree up to depth 3. At each node, it searches every feature and threshold to find the split that minimizes Gini impurity. Returns a nested dictionary — no objects or classes required.

---

## Tech Stack

- Python 3
- NumPy — all algorithm math
- Matplotlib + Seaborn — visualization only
- scikit-learn `confusion_matrix` — evaluation only
- Google Colab compatible

---

## How to Run

**Option 1 — Google Colab (recommended)**

Open the notebook in Colab and run all cells in order.

**Option 2 — Local Jupyter**

```bash
pip install numpy matplotlib seaborn scikit-learn
jupyter notebook fainalML_v6_simple_chapters.ipynb
```

**Sample Dataset**

A ready-to-use dataset is available at:
```
https://drive.google.com/file/d/16AMnMYPwkpWCCO6cZEWzXfUC_fyvnonO/view?usp=sharing
```
Paste this link when the notebook prompts for input.

---

## Who Is This For

- Students learning ML who want to understand algorithms beyond the sklearn API
- Anyone who wants a clean, dependency-light classification baseline they can read and modify
- Courses that require implementing algorithms without using pre-built models

---

## Implementation Notes

- **No data leakage** — scaling parameters are computed on the training set only, then applied to the test set
- **No sklearn for training** — all three models are written from first principles
- **Encoding transparency** — every text-to-integer mapping is stored and printed so the user can trace predictions back to original labels
- **Robust CSV handling** — supports UTF-8 and Latin-1 encodings, empty rows, and inconsistent column widths
