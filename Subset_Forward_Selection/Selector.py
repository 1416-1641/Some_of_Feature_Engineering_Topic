import numpy as np
from utils import clone_estimator
from metrics import get_score


class SequentialFeatureSelector:
    """
    Sequential Feature Selector for feature selection using forward or backward strategy.

    This class selects a subset of features by either:
    - Forward Selection: starting from no features and adding one at a time.
    - Backward Elimination: starting from all features and removing one at a time.

    Attributes:
        estimator: Machine learning model with fit and predict methods.
        n_feature_to_select (int): Number of features to select.
        direction (str): Selection strategy, either 'forward' or 'backward'.
        scoring (str): Evaluation metric to use.
        selected_features (list): Indices of selected features after fitting.
    """

    def __init__(self, estimator, n_features_to_select, direction="forward", scoring="accuracy"):
        """
        Initialize the SequentialFeatureSelector.

        Args:
            estimator: A machine learning model instance.
            n_features_to_select (int): Desired number of features to select.
            direction (str, optional): Selection strategy ('forward' or 'backward').
                Defaults to "forward".
            scoring (str, optional): Scoring metric. Defaults to "accuracy".
        """
        self.estimator = estimator
        self.n_feature_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring

        self.selected_features = None

    def evaluate_subset(self, X, y, feature_indices):
        """
        Evaluate a subset of features using the given estimator.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            feature_indices (list): List of feature indices to evaluate.

        Returns:
            float: Score of the model on the selected feature subset.
        """
        model_cloned = clone_estimator(self.estimator)
        X_subset = X[:, feature_indices]

        model_cloned.fit(X_subset, y)
        y_pred = model_cloned.predict(X_subset)

        score = get_score(y, y_pred, self.scoring)

        return score

    def fit(self, X, y):
        """
        Fit the feature selector to the data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            SequentialFeatureSelector: Returns self for chaining.
        """
        self.n_features = X.shape[1]

        if self.direction == "forward":
            selected = []
            remaining = list(range(self.n_features))

            while len(selected) < self.n_feature_to_select:
                best_score = -np.inf
                best_feature = None

                for f in remaining:
                    current_features = selected + [f]
                    score = self.evaluate_subset(X, y, current_features)

                    if score > best_score:
                        best_score = score
                        best_feature = f

                selected.append(best_feature)
                remaining.remove(best_feature)

            self.selected_features = selected

        elif self.direction == "backward":
            selected = list(range(self.n_features))

            while len(selected) > self.n_feature_to_select:
                best_score = -np.inf
                worst_feature = None

                for f in selected:
                    current_features = [feature for feature in selected if feature != f]
                    score = self.evaluate_subset(X, y, current_features)

                    if score > best_score:
                        best_score = score
                        worst_feature = f

                selected.remove(worst_feature)

            self.selected_features = selected

        return self

    def transform(self, X):
        """
        Transform the dataset by selecting the chosen features.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Transformed feature matrix with selected features only.
        """
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            np.ndarray: Transformed feature matrix.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        """
        Get a boolean mask of selected features.

        Returns:
            list: Boolean list where True indicates selected features.
        """
        mask = [False] * self.n_features

        for idx in self.selected_features:
            mask[idx] = True

        return mask