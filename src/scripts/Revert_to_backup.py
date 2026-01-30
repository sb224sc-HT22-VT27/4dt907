import dagshub
import mlflow
from ML_utils import MLUtils

# Setup dagshub and MLFlow
dagshub.init(repo_owner="SamuelFredricBerg", repo_name="4dt907", mlflow=True)
utils = MLUtils("Project_Model")

utils.revert_backup_to_prod()