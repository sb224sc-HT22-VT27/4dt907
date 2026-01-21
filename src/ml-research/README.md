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

## MLflow Integration

When running notebooks with ML experiments, ensure MLflow tracking server is running:

```bash
# From the src directory
docker-compose up mlflow

# Or run directly
mlflow server --host 0.0.0.0 --port 5000
```

Set the tracking URI in your notebook:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
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
