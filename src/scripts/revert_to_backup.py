import dagshub
from ml_utils import mlutils

# Setup dagshub and MLFlow
dagshub.init(repo_owner="SamuelFredricBerg", repo_name="4dt907", mlflow=True)
utils = mlutils("Project_Model")

utils.revert_backup_to_prod()
