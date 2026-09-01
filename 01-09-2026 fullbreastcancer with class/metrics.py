import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class MetricsEvaluator:
    def evaluate(self, y_test, probabilities):
        results = pd.DataFrame({
            "Actual_class": y_test.values,
            "P_benign": probabilities[:, 0],
            "P_malignant": probabilities[:, 1]
        })

        results["Actual_label"] = results["Actual_class"].map({
            0: "benign",
            1: "malignant"
        })

        print(results.head(10))

        results_table = []

        for threshold in [0.30, 0.50, 0.70, 0.90]:
            predicted_malignant = (
                results["P_malignant"] >= threshold
            ).astype(int)

            print(
                f"Threshold = {threshold}: "
                f"Predicted malignant cases = {predicted_malignant.sum()}"
            )

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                predicted_malignant
            ).ravel()

            accuracy = accuracy_score(
                y_test,
                predicted_malignant
            )

            precision = precision_score(
                y_test,
                predicted_malignant,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predicted_malignant,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predicted_malignant,
                zero_division=0
            )

            results_table.append([
                threshold,
                tn,
                fp,
                fn,
                tp,
                accuracy,
                precision,
                recall,
                f1
            ])

        metrics_table = pd.DataFrame(
            results_table,
            columns=[
                "Threshold",
                "TN",
                "FP",
                "FN",
                "TP",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        )

        display(
            metrics_table.style
            .format({
                "Threshold": "{:.2f}",
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}"
            })
            .set_properties(**{
                "border": "1px solid black",
                "text-align": "center"
            })
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("border", "1px solid black"),
                        ("text-align", "center"),
                        ("font-weight", "bold")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("border", "1px solid black"),
                        ("text-align", "center")
                    ]
                },
                {
                    "selector": "table",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("margin", "10px 0")
                    ]
                }
            ])
        )

        return metrics_table
