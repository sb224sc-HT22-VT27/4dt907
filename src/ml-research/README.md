# ML Research Notebooks

Jupyter notebooks for ML experiments and assignment reporting. Each assignment has its own subdirectory (`a1/`, `a2/`, ...).

## What Has Been Tried

| Assignment | Topic | Notes |
| ------------ | ------- | ------- |
| `a1` | Project setup | Setup of dagshub and utils file |
| `a2` | Regression model setup | Setup baseline for A2 with linear regression, removed outliers using cooks distance. Threshold multiplier 3 was the best.  |
| `a3` | Weak-Link classification setup | Tried RFC, Boosting, Logreg, and ETC. "Best" model from Boosting. |
| `a4` | Comparison of models and improving A2 | Implemented t-test to compare models into utils file aswell as improving A2 using Lasso. |
| `a5` | Ensemble implementation in A2 and A3 | Implemented stacking ensembles to bothe A2 and A3 to whcih improved results |


Models are tracked in MLflow on DagsHub using a utils file, ML_utils.py. 

## Setup

### Prerequisites

- Python 3.12.10
- Jupyter Notebook or JupyterLab

```bash
cd src/ml-research/<current-assignment>
pip install -r requirements.txt
jupyter lab
```

