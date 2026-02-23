# ML Research Notebooks

Jupyter notebooks for ML experiments and assignment reporting. Each assignment has its own subdirectory (`a1/`, `a2/`, ...).

## What Has Been Tried

| Assignment | Topic | Notes |
| ------------ | ------- | ------- |
| `a1` | Project setup & initial data exploration | Dataset: expert wine quality scores |
| `a2` | Baseline regression model (scikit-learn) | Gradient Boosted Trees, logged to MLflow/DagsHub |
| `a3` | Weakest-link analysis | Identified and modelled the weakest-performing feature subset |
| `a4–a16` | Iterative model improvements | Feature engineering, hyperparameter tuning, alternative algorithms |

Models are tracked in MLflow on DagsHub. The best model is promoted to `@prod` via the alias promotion workflow described below.

## Setup

### Prerequisites

- Python 3.12.10
- Jupyter Notebook or JupyterLab

```bash
cd src/ml-research/<current-assignment>
pip install -r requirements.txt
jupyter lab
```

## Promoting a Model to Production

```python
import dagshub, mlflow
from ml_utils import mlutils

dagshub.init(repo_owner="SamuelFredricBerg", repo_name="4dt907", mlflow=True)
utils = mlutils("Project_Model")

utils.promote_dev_to_prod()   # @dev → @prod, @prod → @backup
# utils.revert_backup_to_prod()  # rollback: @backup → @prod
```

Requires `@prod`, `@dev`, and `@backup` aliases to exist on the model in DagsHub.
