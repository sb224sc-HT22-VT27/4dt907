from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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
        print("Attempting to swap @backup to @prod")

        try:
            back_ver = self.client.get_model_version_by_alias(
                self.model_name, "backup"
            ).version
            prod_ver = self.client.get_model_version_by_alias(
                self.model_name, "prod"
            ).version

            self.client.set_registered_model_alias(self.model_name, "prod", back_ver)

            self.client.set_registered_model_alias(self.model_name, "backup", prod_ver)

            print("Success: @backup is now swapped with @prod.")
        except Exception as e:
            print(
                f"An unexpected error occurred during revert (Probably missing some alias): {e}"
            )

    def is_challenger_statistically_better(self, new_scores_df, metric="r2"):
        try:
            results = {"Challenger": new_scores_df[metric].values}
            aliases = [("Prod", "prod"), ("Dev", "dev")]

            for label, alias in aliases:
                try:
                    ver = self.client.get_model_version_by_alias(self.model_name, alias)
                    path = self.client.download_artifacts(
                        ver.run_id, "cv_fold_scores.csv", "."
                    )
                    results[label] = pd.read_csv(path)[metric].values
                except Exception:
                    pass

            plt.figure(figsize=(10, 5))
            colors = {"Challenger": "blue", "Prod": "green", "Dev": "red"}
            all_vals = np.concatenate(list(results.values()))
            x_axis = np.linspace(all_vals.min() - 0.05, all_vals.max() + 0.05, 500)

            for label, scores in results.items():
                mu, se = np.mean(scores), stats.sem(scores)
                pdf = stats.t.pdf(x_axis, len(scores) - 1, loc=mu, scale=se)  # v = k-1
                plt.plot(
                    x_axis,
                    pdf,
                    label=f"{label} $\mu$={mu:.4f}",
                    color=colors.get(label, "black"),
                    lw=2,
                )
                plt.fill_between(
                    x_axis, pdf, alpha=0.1, color=colors.get(label, "black")
                )

            plt.title("Statistical Confidence Comparison (PDFs)")
            plt.xlabel(f"{metric.upper()} Score")
            plt.ylabel("Probability Density")
            plt.legend()
            plt.savefig("model_confidence_comparison.png")
            plt.show()

            if "Prod" in results:
                prod_scores = results["Prod"]
                new_scores = results["Challenger"]
                k = len(new_scores)
                n_test, n_train = 0.2, 0.8  #

                diffs = new_scores - prod_scores
                d_bar = np.mean(diffs)
                s_sq = np.var(diffs, ddof=1)

                var_corrected = ((1 / k) + (n_test / n_train)) * s_sq
                t_stat = d_bar / np.sqrt(var_corrected)  #
                p_val = stats.t.sf(np.abs(t_stat), k - 1) * 2  #

                se_corr = np.sqrt(var_corrected)
                x_diff = np.linspace(d_bar - 4 * se_corr, d_bar + 4 * se_corr, 500)
                p_x = stats.t.pdf(x_diff, k - 1, loc=d_bar, scale=se_corr)

                plt.figure(figsize=(10, 5))
                plt.plot(
                    x_diff, p_x, label="Difference Distribution", color="purple", lw=2
                )
                plt.fill_between(x_diff, 0, p_x, alpha=0.1, color="purple")
                plt.axvline(
                    0, color="red", linestyle="--", label="Null Hypothesis ($x=0$)"
                )
                plt.axvline(
                    d_bar, color="green", linestyle=":", label=f"Mean Diff: {d_bar:.4f}"
                )

                plt.title(
                    f"corrected Resampled t-Test (Challenger - Prod)\np-value: {p_val:.4f}"
                )
                plt.xlabel("Difference (x)")
                plt.ylabel("P(x)")
                plt.legend()
                plt.savefig("challenger_vs_prod_ttest.png")
                plt.show()

                return (d_bar > 0 and p_val < 0.05), p_val

            return True, 1.0

        except Exception as e:
            print(f"Error: {e}")
            return True, 1.0
