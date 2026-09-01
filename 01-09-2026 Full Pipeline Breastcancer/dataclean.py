# Data Loading & Cleaning

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    (data.target == 0).astype(int),
    name="malignant"
)

print(y.value_counts())

print("Feature matrix shape: ", X.shape)
print("Target shape: ", y.shape)
print("Class names: ", data.target_names)

class_counts = y.value_counts().sort_index()

class_distribution = pd.DataFrame(
    {
        "Class": data.target_names,
        "Count": class_counts.values,
        "Probability": class_counts.values / len(y)
    }
)

print(class_distribution)

class_distribution.plot(
    x="Class",
    y="Count",
    kind="bar",
    legend=False,
    color=["orange", "yellow"]
)

plt.ylabel("Number of observations")
plt.title("Class Distribution")
plt.xticks(rotation=0)
plt.show()
