# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training size: ", len(y_train))
print("Testing size: ", len(y_test))

print("\nTraining Proportions: ")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTesting Proportions: ")
print(y_test.value_counts(normalize=True).sort_index())


# Model Training

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)

print(probabilities[:5])


# Results

results = pd.DataFrame({
    "Actual_class": y_test.values,
    "P_malignant": probabilities[:, 0],
    "P_benign": probabilities[:, 1]
})

print(results.head(10))

results["Actual_label"] = results["Actual_class"].map({
    0: "benign",
    1: "malignant"
})

print(
    results[
        [
            "Actual_class",
            "Actual_label",
            "P_malignant",
            "P_benign"
        ]
    ].head(10)
)
