import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class ModelTrainer:
    def split_data(self, X, y):
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

        return X_train, X_test, y_train, y_test

    def train_and_predict(self, X_train, X_test, y_train):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)

        print(probabilities[:5])

        return probabilities
