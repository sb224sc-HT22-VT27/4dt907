# ML Research Notebooks

Jupyter notebooks for ML experiments and assignment reporting. Each assignment has its own subdirectory (`a1/`, `a2/`, ...).

## What Has Been Tried

| Assignment | Topic | Notes |
| ------------ | ------- | ------- |
| $A1$ | Project setup | Setup of dagshub and utils file |
| $A2$ | Regression model setup | Setup baseline for A2 with linear regression, removed outliers using cooks distance. Threshold multiplier 3 was the best. |
| $A3$ | Weak-Link classification setup | Tried RFC, Boosting, Logreg, and ETC. "Best" model from Boosting. |
| $A4$ | Comparison of models and improving A2 | Implemented t-test to compare models into utils file aswell as improving A2 using Lasso. |
| $A5$ | Ensemble implementation in A2 and A3 | Implemented stacking ensembles to bothe A2 and A3 to whcih improved results |
| $A5b$ | Hyperparameter tuning | Tested different hyperparameters to try and improve results |
| $A6$ | - | - |
| $A7$ | - | - |
| $A8$ | - | - |
| $A9$ | - | - |
| $A10$ | - | - |
| $A11$ | - | - |
| $A12$ | - | - |
| $A13$ | - | - |
| $A14$ | - | - |
| $A15$ | - | - |
| $A16$ | - | - |
| PoseNet | Adapt the PoseNet Python port | Run PoseNet on images/videos; joint positions saved to CSV and JSON files |

Models are tracked in MLflow on DagsHub using a utils file, ml_utils.py.

## Setup

### Prerequisites

- Python 3.12.10
- Jupyter Notebook or JupyterLab

```bash
cd src/ml-research/<current-assignment>
pip install -r requirements.txt
jupyter lab
```

### PoseNet assignment

```bash
cd src/ml-research/posenet
pip install -r requirements.txt
jupyter lab posenet_demo.ipynb
```

Place input images in `src/ml-research/posenet/images/` and a video file at  
`src/ml-research/posenet/video.mp4` before running the notebook.  
The notebook will write joint-position data to `src/ml-research/posenet/output/`.
