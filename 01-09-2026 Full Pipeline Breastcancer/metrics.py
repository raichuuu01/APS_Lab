# Metrics

actual_malignant = (y_test.values == 0).astype(int)

results_table = []

for threshold in [0.30, 0.50, 0.70, 0.90]:

    predicted_malignant = (
        probabilities[:, 0] >= threshold
    ).astype(int)

    print(
        f"Threshold = {threshold}: "
        f"Predicted malignant cases = {predicted_malignant.sum()}"
    )

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(
        actual_malignant,
        predicted_malignant
    ).ravel()

    # Metrics
    accuracy = accuracy_score(
        actual_malignant,
        predicted_malignant
    )

    precision = precision_score(
        actual_malignant,
        predicted_malignant,
        zero_division=0
    )

    recall = recall_score(
        actual_malignant,
        predicted_malignant,
        zero_division=0
    )

    f1 = f1_score(
        actual_malignant,
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


# Metrics Table

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
