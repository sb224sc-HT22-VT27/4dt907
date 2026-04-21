# ML Research Notebooks

Jupyter notebooks for ML experiments and assignment reporting. Each assignment has its own subdirectory (`a1/`, `a2/`, ...).

## What Has Been Tried

| Assignment | Topic | Notes |
| ------------ | ------- | ------- |
| $A1$ | Project setup | Setup of dagshub and utils file |
| $A2$ | Regression model setup | Setup baseline for A2 with linear regression, removed outliers using cooks distance. Threshold multiplier 3 was the best. |
| $A3$ | Weak-Link classification setup | Tried RFC, Boosting, Logreg, and ETC. "Best" model from Boosting. |
| $A4$ | Comparison of models and improving regreesion model | Implemented t-test to compare models into utils file aswell as improving A2 using Lasso. |
| $A5$ | Ensemble implementation in regression and weak link | Implemented stacking ensembles to both to which improved results |
| $A5b$ | Hyperparameter tuning | Tested different hyperparameters to try and improve results |
| $A6$ | SVM grid optimization | Weaklink level0 implementation of new optimized parameters |
| $A7$ | PCA and Clustering | Did PCA and clustering on weak link data |
| $A8$ | Preparation for DL | Prepared for DL by reading docs and tutorials |
| $A9$ | DeepLearning model to predict z | Using kinectdata transformed to mediapipe to train a model to predict z form x,y |
| $A10$ | - | - |
| $A11$ | - | - |
| $A12$ | - | - |
| $A13$ | - | - |
| $A14$ | - | - |
| $A15$ | - | - |
| $A16$ | - | - |

Models are tracked in MLflow on DagsHub using a utils file, ml_utils.py.

### Models
A2 (Regression Model) - Regression model to predict aimoscore
A3 (Weaklink Model) - Model that predicts th weakest link
A9 (z Prediction Model) - Model that predicts z from x,y

## Setup

### Prerequisites

- Python 3.12.10
- Jupyter Notebook or JupyterLab

```bash
cd src/ml-research/<current-assignment>
pip install -r requirements.txt
jupyter lab
```
