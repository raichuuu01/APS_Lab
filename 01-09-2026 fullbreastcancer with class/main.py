from dataclean import DataCleaner
from modeltrain import ModelTrainer
from metrics import MetricsEvaluator


class Main:
    def run(self):
        cleaner = DataCleaner()
        X, y = cleaner.load_and_prepare_data()

        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        probabilities = trainer.train_and_predict(X_train, X_test, y_train)

        evaluator = MetricsEvaluator()
        evaluator.evaluate(y_test, probabilities)


if __name__ == "__main__":
    Main().run()
