from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%

    # Normalization to [0, 1]
    # X_max, X_min = np.max(X, axis=0), np.min(X, axis=0)
    # for i in range(X.shape[1]):
    #     X[:, i] = (X[:, i] - X_min[i]) / (X_max[i] - X_min[i])
    
    # Standardization
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = (X[i, j] - X_mean[j]) / X_std[j]
    
    return X


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_map[label] for label in y])
    return encoded_labels


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO: 2%
        X = np.insert(X, 0, 1, axis=1)
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        if self.model_type == "logistic":
            self.weights = np.zeros((n_features, n_classes))
            
            for _ in range(self.iterations):
                gradients = self._compute_gradients(X, y)
                self.weights -= self.learning_rate * gradients
        else:
            self.weights = np.zeros(n_features)
            for _ in range(self.iterations):
                gradients = self._compute_gradients(X, y)
                self.weights -= self.learning_rate * gradients

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            y_pred = X @ self.weights
            return y_pred
        elif self.model_type == "logistic":
            # TODO: 2
            y_pred = self._softmax(X @ self.weights)
            return np.argmax(y_pred, axis=1)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_classes = len(np.unique(y))
        n_samples, n_features = X.shape

        if self.model_type == "linear":
            # TODO: 2%
            y_pred = X @ self.weights
            gradients = -X.T @ (y - y_pred) / n_samples
            return gradients
        elif self.model_type == "logistic":
            # TODO: 2%
            y_one_hot = np.zeros((n_samples, n_classes))
            for i in range(n_samples):
                y_one_hot[i, y[i]] = 1
            
            y_pred = self._softmax(X @ self.weights)
            gradients = -X.T @ (y_one_hot - y_pred) / n_samples
            return gradients

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        mask = X[:, feature] <= threshold
        left_child = self._build_tree(X[mask], y[mask], depth + 1)
        right_child = self._build_tree(X[~mask], y[~mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            counts = Counter(y)
            return counts.most_common(1)[0][0]
        else:
            # TODO: 1%
            return np.mean(y)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]
                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 2%
        def gini(y: np.ndarray) -> float:
            if len(y) == 0:
                return 0
            counts = Counter(y)
            probs = np.array(list(counts.values())) / len(y)
            return 1 - np.sum(probs ** 2)
            
        n_samples = len(left_y) + len(right_y)
        gini_left, gini_right = gini(left_y), gini(right_y)
        
        return (len(left_y) * gini_left + len(right_y) * gini_right) / n_samples

    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 2%
        def mse(y: np.ndarray) -> float:
            if len(y) == 0:
                return 0
            return np.mean((y - np.mean(y)) ** 2)
        n_samples = len(left_y) + len(right_y)
        mse_left, mse_right = mse(left_y), mse(right_y)
        return (len(left_y) * mse_left + len(right_y) * mse_right) / n_samples

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.model_type = model_type
        self.trees = [DecisionTree(max_depth=max_depth, model_type=model_type) for _ in range(n_estimators)]
        
        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        for tree in self.trees:
            # TODO: 2%
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            tree.fit(X_bootstrap, y_bootstrap)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        n_samples = X.shape[0]
        if self.model_type == "classifier":
            predictions = []
            for tree in self.trees:
                predictions.append(tree.predict(X))
            
            predictions = np.array(predictions).T
            return np.array([Counter(row).most_common(1)[0][0] for row in predictions])
        else:
            predictions = np.zeros(n_samples)
            for tree in self.trees:
                predictions += tree.predict(X)
            return predictions / len(self.trees)

# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    return np.mean((y_true - y_pred) ** 2)


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    # logistic_regression = LinearModel(learning_rate=0.1, iterations=10000, model_type="logistic")
    logistic_regression = LinearModel(learning_rate=0.1 , model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    # decision_tree_classifier = DecisionTree(max_depth=5, model_type="classifier")
    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    # random_forest_classifier = RandomForest(n_estimators=1, max_depth=5, model_type="classifier")
    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(learning_rate=0.02, iterations=1000, model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(max_depth=6, model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(n_estimators=150, max_depth=5, model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()