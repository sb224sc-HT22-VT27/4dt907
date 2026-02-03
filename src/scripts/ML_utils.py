import logging
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


class MLUtils:
    def __init__(self, model_name="Project_Model"):
        self.client = MlflowClient()
        self.model_name = model_name
        logger.info(f"Initialized MLUtils for model: {model_name}")

    def auto_check_challenger(self, run_id, metric_name="accuracy"):
        """
        Function to check if new model should be saved or not depending on accuracy.
        Returns true if new model beat @dev
        Returns false if new model does not beat @dev
        """
        logger.info(f"Checking if new model (run_id: {run_id}) beats current @dev")
        
        try:
            new_run = self.client.get_run(run_id)
            new_score = new_run.data.metrics.get(metric_name, 0)
            logger.info(f"New model {metric_name}: {new_score}")
        except MlflowException as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            raise
        
        try:
            # Get current @dev and @backup versions
            dev_ver = self.client.get_model_version_by_alias(self.model_name, "dev")
            dev_score = self.client.get_run(dev_ver.run_id).data.metrics.get(metric_name, 0)
            logger.info(f"Current @dev model {metric_name}: {dev_score}")
        except MlflowException as e:
            # If no dev exists
            logger.info(f"No @dev model exists yet: {e}")
            dev_score = -1

        if new_score > dev_score:
            logger.info(f"New Best! {new_score} > {dev_score}. Updating models on DagsHub")
            print(f"New Best! {new_score} > {dev_score}. Updating models on DagsHub")
            
            # Delete current @backup
            self._delete_current_backup()

            # Move current @dev -> @backup
            try:
                old_dev = self.client.get_model_version_by_alias(self.model_name, "dev")
                logger.info(f"Moving @dev (version {old_dev.version}) to @backup")
                self.client.set_registered_model_alias(self.model_name, "backup", old_dev.version)
                # Remove @dev alias from the old version
                self.client.delete_registered_model_alias(self.model_name, "dev")
                logger.info(f"Successfully moved @dev to @backup")
            except MlflowException as e:
                # No dev existed yet
                logger.info(f"No existing @dev to move to @backup: {e}")

            return True
        
        logger.info(f"New model ({new_score}) did not beat @dev ({dev_score})")
        return False

    def _delete_current_backup(self):
        """Finds the version with @backup deletes the model."""
        try:
            backup_ver = self.client.get_model_version_by_alias(self.model_name, "backup")
            logger.info(f"Found @backup model version: {backup_ver.version}")
            
            # Safety Check so it does not accidentally delete @prod
            try:
                prod_v = self.client.get_model_version_by_alias(self.model_name, "prod").version
                logger.info(f"Current @prod model version: {prod_v}")
            except MlflowException as e:
                logger.info(f"No @prod model exists: {e}")
                prod_v = None

            if backup_ver.version != prod_v:
                logger.info(f"Deleting old backup model version {backup_ver.version}")
                print(f"Deleting old backup model.")
                self.client.delete_model_version(self.model_name, backup_ver.version)
                logger.info(f"Successfully deleted backup version {backup_ver.version}")
            else:
                logger.warning(f"@backup (version {backup_ver.version}) is same as @prod. Not deleting.")
                print(f"Warning: @backup is @prod. Not deleting.")
        except MlflowException as e:
            logger.info(f"No backup model to delete: {e}")

    def promote_dev_to_prod(self):
        """
        Rotates aliases: 
        Current @prod becomes @backup
        Current @dev becomes @prod
        Current @backup becomes @dev
        """
        logger.info("Attempting to promote @dev into @prod")
        print("Attempting to promote @dev into @prod")

        try:
            dev_ver = self.client.get_model_version_by_alias(self.model_name, "dev").version
            prod_ver = self.client.get_model_version_by_alias(self.model_name, "prod").version
            back_ver = self.client.get_model_version_by_alias(self.model_name, "backup").version

            logger.info(f"Current aliases - @dev: {dev_ver}, @prod: {prod_ver}, @backup: {back_ver}")

            self.client.set_registered_model_alias(self.model_name, "backup", prod_ver)
            self.client.set_registered_model_alias(self.model_name, "prod", dev_ver)
            self.client.set_registered_model_alias(self.model_name, "dev", back_ver)

            logger.info(f"Promotion complete - New aliases: @prod: {dev_ver}, @backup: {prod_ver}, @dev: {back_ver}")
            print("Promotion complete! Following swaps have been made:")
            print("@dev -> @prod")
            print("@prod -> @backup")
            print("@backup -> @dev")
        except MlflowException as e:
            logger.error(f"Promotion failed: {e}")
            print(f"Promotion failed: Not all models exist. Error: {e}")
    
    def revert_backup_to_prod(self):
        """
        Swaps @prod and @backup.
        Used if the current @prod is buggy and you need to restore the previous stable version.
        """
        logger.info("Attempting to swap @backup to @prod")
        print(f"Attempting to swap @backup to @prod")

        try:
            back_ver = self.client.get_model_version_by_alias(self.model_name, "backup").version
            prod_ver = self.client.get_model_version_by_alias(self.model_name, "prod").version

            logger.info(f"Current @backup: {back_ver}, Current @prod: {prod_ver}")

            self.client.set_registered_model_alias(self.model_name, "prod", back_ver)
            self.client.set_registered_model_alias(self.model_name, "backup", prod_ver)

            logger.info(f"Successfully swapped - New @prod: {back_ver}, New @backup: {prod_ver}")
            print(f"Success: @backup is now swapped with @prod.")
        except MlflowException as e:
            logger.error(f"Revert failed: {e}")
            print(f"An unexpected error occurred during revert (Probably missing some alias): {e}")