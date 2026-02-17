import dagshub
import mlflow
from ml_utils import MLUtils

# Setup dagshub and MLFlow
dagshub.init(repo_owner="SamuelFredricBerg", repo_name="4dt907", mlflow=True)
utils = MLUtils("Project_Model")

utils.promote_dev_to_prod()
