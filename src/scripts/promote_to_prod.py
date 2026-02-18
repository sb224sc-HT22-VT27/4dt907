import dagshub
from ml_utils import mlutils

# Setup dagshub and MLFlow
dagshub.init(repo_owner="SamuelFredricBerg", repo_name="4dt907", mlflow=True)
utils = mlutils("Project_Model")

utils.promote_dev_to_prod()
