from mlflow.tracking import MlflowClient


class MLUtils:
    def __init__(self, model_name="Project_Model"):
        self.client = MlflowClient()
        self.model_name = model_name

    def auto_check_challenger(self, run_id, metric_name="R2_Mean"):
        """
        Function to check if new model should be saved or not depending on R2_Mean.
        Returns true if new model beat @dev
        Returns false if new model does not beat @dev
        """
        new_run = self.client.get_run(run_id)
        new_score = new_run.data.metrics.get(metric_name, 0)

        try:
            # Get current @dev and @backup versions
            dev_ver = self.client.get_model_version_by_alias(self.model_name, "dev")
            dev_score = self.client.get_run(dev_ver.run_id).data.metrics.get(
                metric_name, 0
            )
        except Exception:
            # If no dev exists
            dev_score = -1

        if new_score > dev_score:
            print(f"New Best! {new_score} > {dev_score}. Updating models on DagsHub")

            # Delete current @backup
            self._delete_current_backup()

            # Move current @dev -> @backup
            try:
                old_dev = self.client.get_model_version_by_alias(self.model_name, "dev")
                self.client.set_registered_model_alias(
                    self.model_name, "backup", old_dev.version
                )
                # Remove @dev alias from the old version
                self.client.delete_registered_model_alias(self.model_name, "dev")
            except BaseException:
                # No dev existed yet
                pass

            return True

        return False

    def _delete_current_backup(self):
        """Finds the version with @backup deletes the model."""
        try:
            backup_ver = self.client.get_model_version_by_alias(
                self.model_name, "backup"
            )

            # Safety Check so it does not accidentally delete @prod
            try:
                prod_v = self.client.get_model_version_by_alias(
                    self.model_name, "prod"
                ).version
            except BaseException:
                prod_v = None

            if backup_ver.version != prod_v:
                print("Deleting old backup model.")
                self.client.delete_model_version(self.model_name, backup_ver.version)
            else:
                print("Warning: @backup is @prod. Not deleting.")
        except BaseException:
            pass  # No backup existed to delete

    def promote_dev_to_prod(self):
        """
        Rotates aliases:
        Current @prod becomes @backup
        Current @dev becomes @prod
        Current @backup becomes @dev
        """

        print("Attempting to promote @dev into @prod")

        try:
            dev_ver = self.client.get_model_version_by_alias(
                self.model_name, "dev"
            ).version
            prod_ver = self.client.get_model_version_by_alias(
                self.model_name, "prod"
            ).version
            back_ver = self.client.get_model_version_by_alias(
                self.model_name, "backup"
            ).version

            self.client.set_registered_model_alias(self.model_name, "backup", prod_ver)
            self.client.set_registered_model_alias(self.model_name, "prod", dev_ver)
            self.client.set_registered_model_alias(self.model_name, "dev", back_ver)

            print("Promotion complete! Following swaps have been made:")
            print("@dev -> @prod")
            print("@prod -> @backup")
            print("@backup -> @dev")
        except Exception as e:
            print(f"Promotion failed: Not all models exist. Error: {e}")

    def revert_backup_to_prod(self):
        """
        Swaps @prod and @backup.
        Used if the current @prod is buggy and you need to restore the previous stable version.
        """
        print(f"Attempting to swap @backup to @prod")

        try:
            back_ver = self.client.get_model_version_by_alias(
                self.model_name, "backup"
            ).version
            prod_ver = self.client.get_model_version_by_alias(
                self.model_name, "prod"
            ).version

            self.client.set_registered_model_alias(self.model_name, "prod", back_ver)

            self.client.set_registered_model_alias(self.model_name, "backup", prod_ver)

            print(f"Success: @backup is now swapped with @prod.")
        except Exception as e:
            print(
                f"An unexpected error occurred during revert (Probably missing some alias): {e}"
            )
