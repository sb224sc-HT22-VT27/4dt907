# ML Research Notebooks

This directory contains Jupyter notebooks for ML research and assignment reporting.

## Structure

Each assignment has its own subdirectory:

- `a1/` - Assignment 1: Project Setup
- `a2/` - Assignment 2
- `a3/` - Assignment 3
- ... and so on

## Setup

### Prerequisites

- Python 3.12.x
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook

# Or start JupyterLab
jupyter lab
```

## Running Notebooks

1. Navigate to the assignment directory (e.g., `a1/`)
2. Open the corresponding notebook file (e.g., `a1.ipynb`)
3. Run the cells sequentially

## Creating New Notebooks
1. Create a notebook cell containing following information:
```python
import dagshub
import mlflow
import scripts.ML_utils as MLUtils

# Setup dagshub and MLFlow
dagshub.init(repo_owner="SamuelFredricBerg", repo_name="4dt907", mlflow=True)
utils = MLUtils("Project_Model")


# Add all configs used in the training (EXAMPLE BELOW)
config = {
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.05,
    "feature_set": "v2_processed",
    "data_split_seed": 42
}
```
2. Main cell should look like this.
```python
with mlflow.start_run(run_name="ADD_RUN_NAME_HERE") as run:
    mlflow.log_params(config)


    # TRAINING CODE GOES HERE
    # model = train_model()

    # Logging results
    # Create a dictionary for all results (EXAMPLE BELOW)
    results = {
        "accuracy": 0.942,
        "f1_score": 0.915,
        "precision": 0.920,
        "recall": 0.910
    }
    mlflow.log_metrics(results)

    # Logging visuals
    # Example how to below
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    if utils.auto_check_challenger(run.info.run_id, metric_name="accuracy"):
        mlflow.sklearn.log_model(model, "model", registered_model_name="Project_Model")

        latest_v = utils.client.get_latest_versions("Project_Model")[0].version
        utils.client.set_registered_model_alias("Project_Model", "dev", latest_v)
        print("New model beat current @dev uploading to DagsHub")
    else:
        print("Did not beat current @dev, model not uploaded to Dagshub")
```
3. Get training data
The training data is located at ...

## Production Model Update
Prerequisites of doing a manual update is that all three aliases need to exist inside the model on DagsHub. If a new model name is used ensure to have @prod, @dev and @backup on three different models. If there are only two models additional setup has to be made inside DagsHub.
Ways to fix:
1. Manually force a model to be uploaded to the model and set that to missing alias.
2. Check if there exist another model that can be manually promoted to the current one and set the missing alias.

### Manual Update
Run the following code to change @dev model to @prod as well as @backup to @dev.
```python
import dagshub
import mlflow
from ML_utils import MLUtils

# Setup dagshub and MLFlow
dagshub.init(repo_owner="SamuelFredricBerg", repo_name="4dt907", mlflow=True)
utils = MLUtils("Project_Model")

utils.promote_dev_to_prod()
```

### Incase of @prod problems
Incase the new @prod model does not work we can change place of @backup and @prod reverting to the previous @prod model.

```python
import dagshub
import mlflow
from ML_utils import MLUtils

# Setup dagshub and MLFlow
dagshub.init(repo_owner="SamuelFredricBerg", repo_name="4dt907", mlflow=True)
utils = MLUtils("Project_Model")

utils.revert_backup_to_prod()
```

## Best Practices

1. **Documentation**: Include markdown cells explaining your analysis
2. **Reproducibility**: Set random seeds for reproducible results
3. **Code Quality**: Follow PEP 8 style guidelines
4. **Version Control**: Do NOT commit large data files or model artifacts
5. **Clean Outputs**: Consider clearing outputs before committing notebooks

## Dependencies

See `requirements.txt` for the full list of dependencies.

Key libraries:

- **Jupyter**: Interactive notebook environment
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **scikit-learn**: Machine learning library
- **MLflow**: ML experiment tracking
- **dagshub**: DagsHub model tracking

## Contributing

When adding new notebooks:

1. Create a new directory for each major assignment
2. Use descriptive names for notebooks
3. Include a brief description at the top of each notebook
4. Document your process and findings
5. Clear cell outputs if they contain sensitive data

## Issues

If you encounter issues:

1. Check that all dependencies are installed
2. Verify Jupyter is properly configured
3. Ensure Python version is 3.12.x
4. Report issues in the project repository
