"""
train.py
--------
Trains a Random Forest classifier for network intrusion detection and
reports real evaluation metrics (accuracy, precision, recall, F1,
confusion matrix) on a held-out test set.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def load_data(path="data/network_traffic.csv"):
    """
    Load the dataset and split it into features and target.

    Args:
        path: Location of the CSV file containing the network traffic data.

    Returns:
        X: Feature matrix with all columns except the label.
        y: Target label column indicating normal vs attack traffic.
    """
    df = pd.read_csv(path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def train_and_evaluate():
    """
    Train a Random Forest classifier and evaluate its performance.

    The function:
    - loads the dataset,
    - splits it into train/test sets,
    - trains a Random Forest model,
    - measures classification performance,
    - prints evaluation results,
    - saves the trained model for later use in prediction.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create the classifier and train it on the training split.
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Make predictions on unseen test data.
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("=== Evaluation on held-out test set (20% of data, unseen during training) ===")
    for name, val in metrics.items():
        print(f"{name.capitalize():10s}: {val:.1%}")

    print("\nConfusion matrix (rows=actual, cols=predicted) [normal, attack]:")
    print(confusion_matrix(y_test, y_pred))

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

    # Feature importance — useful for explaining *why* the model flags a flow
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top 5 most important features:")
    print(importances.head(5))

    # Save the trained model so it can be reused later for prediction.
    joblib.dump(model, "src/model.joblib")
    print("\nModel saved to src/model.joblib")

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
