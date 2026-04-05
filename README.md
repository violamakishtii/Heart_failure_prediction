# Heart Disease Classification — Machine Learning Coursework

A machine learning project that applies four classification algorithms to predict heart disease using the UCI Heart Failure dataset.

---

## Dataset — `heart.csv`

| Property | Value |
|---|---|
| Rows | 918 |
| Columns | 12 |
| Target | `HeartDisease` (0 = No, 1 = Yes) |
| Source | UCI Heart Failure Prediction Dataset |

### Features

| Column | Type | Description |
|---|---|---|
| `Age` | int | Age of the patient (years) |
| `Sex` | str | Sex of the patient (M / F) |
| `ChestPainType` | str | Chest pain type: ATA, NAP, ASY, TA |
| `RestingBP` | int | Resting blood pressure (mm Hg) |
| `Cholesterol` | int | Serum cholesterol (mg/dl) |
| `FastingBS` | int | Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No) |
| `RestingECG` | str | Resting ECG results: Normal, ST, LVH |
| `MaxHR` | int | Maximum heart rate achieved |
| `ExerciseAngina` | str | Exercise-induced angina (Y / N) |
| `Oldpeak` | float | ST depression induced by exercise |
| `ST_Slope` | str | Slope of peak exercise ST segment: Up, Flat, Down |
| `HeartDisease` | int | Target — 0: No disease, 1: Disease |

### Target Distribution
- **No Heart Disease (0):** 410 patients (44.7%)
- **Heart Disease (1):** 508 patients (55.3%)

---

## Project Structure

```
Heart-Labwork/
│
├── heart.csv                  # Dataset
├── heart_disease_ml.ipynb     # Main Jupyter notebook
└── README.md                  # This file
```

---

## Steps Covered (2–8)

| Step | Description |
|---|---|
| 2 | Load and explore the dataset, encode categorical features, train/test split |
| 3 | Decision Tree Classifier — tree plot, feature importances, confusion matrix |
| 4 | Linear Regression — thresholded at 0.5 for binary classification |
| 5 | Logistic Regression — confusion matrix, ROC curve, feature coefficients |
| 6 | Naive Bayes (GaussianNB) — confusion matrix |
| 7 | ROC curves for all models on one plot |
| 8 | Summary table and bar chart comparing all models |

---

## How to Run

### In JupyterLite / Pyodide (browser-based)

Paste and run this single cell — it installs all dependencies and loads the data automatically:

```python
import micropip
await micropip.install(['seaborn', 'scikit-learn'])

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve,
                              roc_auc_score, f1_score, precision_score, recall_score)

from pyodide.http import pyfetch
response = await pyfetch("https://raw.githubusercontent.com/violamakishtii/Heart-Labwork/main/heart.csv")
with open("heart.csv", "wb") as f:
    f.write(await response.bytes())

df = pd.read_csv('heart.csv')
display(df.head())
print('Shape:', df.shape)

df_encoded = pd.get_dummies(df, columns=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'], drop_first=True)
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

plt.figure(figsize=(24,8))
plot_tree(dt, feature_names=X.columns.tolist(), class_names=['No','Yes'], filled=True, rounded=True, fontsize=8)
plt.title('Decision Tree'); plt.show()

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree'); plt.show()

lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr_raw = lr.predict(X_test_s)
y_pred_lr = (y_pred_lr_raw >= 0.5).astype(int)
print("Linear Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix - Linear Regression'); plt.show()

log = LogisticRegression(max_iter=1000, random_state=42)
log.fit(X_train_s, y_train)
y_pred_log = log.predict(X_test_s)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Logistic Regression'); plt.show()

nb = GaussianNB()
nb.fit(X_train_s, y_train)
y_pred_nb = nb.predict(X_test_s)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix - Naive Bayes'); plt.show()

plt.figure(figsize=(8,6))
for name, proba, color in [
    ('Decision Tree',       dt.predict_proba(X_test)[:,1],    'steelblue'),
    ('Linear Regression',   y_pred_lr_raw,                    'salmon'),
    ('Logistic Regression', log.predict_proba(X_test_s)[:,1], 'green'),
    ('Naive Bayes',         nb.predict_proba(X_test_s)[:,1],  'orange'),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, proba):.3f})', color=color)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curves'); plt.legend(); plt.show()

results = pd.DataFrame({
    'Model':     ['Decision Tree', 'Linear Regression', 'Logistic Regression', 'Naive Bayes'],
    'Accuracy':  [accuracy_score(y_test, p)  for p in [y_pred_dt, y_pred_lr, y_pred_log, y_pred_nb]],
    'Precision': [precision_score(y_test, p) for p in [y_pred_dt, y_pred_lr, y_pred_log, y_pred_nb]],
    'Recall':    [recall_score(y_test, p)    for p in [y_pred_dt, y_pred_lr, y_pred_log, y_pred_nb]],
    'F1':        [f1_score(y_test, p)        for p in [y_pred_dt, y_pred_lr, y_pred_log, y_pred_nb]],
}).set_index('Model').round(4)

display(results)
results.plot(kind='bar', figsize=(10,5), colormap='Set2', edgecolor='black', ylim=(0.6,1.0))
plt.title('Model Comparison'); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.show()
```

### In Classic Jupyter Notebook / Google Colab

Replace the `pyfetch` block with:
```python
df = pd.read_csv('heart.csv')   # place heart.csv in the same folder
```
And remove the `await micropip.install(...)` line — libraries are pre-installed.

---

## Results Summary

| Model | Accuracy | Notes |
|---|---|---|
| Decision Tree | ~83.7% | No scaling needed. Interpretable. |
| Linear Regression | ~89.1% | Not a classifier — output thresholded at 0.5. |
| Logistic Regression | ~88.6% | Best for probability estimates. |
| Naive Bayes | ~91.3% | Best overall accuracy on this dataset. |

---

## Push to GitHub

```bash
git init
git remote add origin https://github.com/violamakishtii/Heart-Labwork.git
git add heart.csv heart_disease_ml.ipynb README.md
git commit -m "Add heart disease ML notebook and README"
git push -u origin main
```

---

## Notes

- All categorical columns are one-hot encoded with `drop_first=True`
- Train/test split: **80% / 20%** with `stratify=y` to preserve class balance
- Scaling with `StandardScaler` is applied only for Linear Regression, Logistic Regression, and Naive Bayes — Decision Trees do not require it
- `random_state=42` is used throughout for reproducibility
