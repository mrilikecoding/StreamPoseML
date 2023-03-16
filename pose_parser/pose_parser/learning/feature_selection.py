import pandas as pd
import pickle
import time

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# TODO this can just be a jupyter notebook - just getting a rough and dirty model trained to start


def build_random_forest(data_file):
    """
    This
    """
    data_file = data_file
    X = pd.read_csv(data_file, index_col=0)
    X.drop(["video_id"], axis=1, inplace=True)
    X.dropna(thresh=X.shape[1], inplace=True)
    y = X.weight_transfer_type
    X.drop(["weight_transfer_type"], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=10, cv=5
    )

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print("Best hyperparameters:", rand_search.best_params_)

    # Generate predictions with the best model
    y_pred = best_rf.predict(X_test)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("confusion matrix")
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # Create a series containing feature importances from the model and feature names from the training data
    feature_importances = pd.Series(
        best_rf.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    print("Feature importance")
    print(feature_importances)
    test_results = {
        "type": "Random Forest",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "classifier": best_rf,
        "X_train": X_train,
        "X_test": X_train,
        "y_train": y_train,
        "y_test": y_test,
    }
    return test_results


if __name__ == "__main__":
    data_file = "./data/annotated_videos/dataset_1678732901064497000.csv"
    results = build_random_forest(data_file=data_file)
    saved_model_path = f"./data/models/random-forest-{time.time_ns()}.pickle"
    if True:
        # Todo fix mkdir issue
        with open(saved_model_path, "wb") as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        print("Saved model to pickle!")
