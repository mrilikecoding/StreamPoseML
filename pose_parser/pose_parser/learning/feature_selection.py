import pandas as pd
import pickle
import time
import json

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


def build_random_forest(data_file: str, column_whitelist: list = []):
    data_file = data_file
    X = pd.read_csv(data_file, index_col=0)

    # convert string values
    X.weight_transfer_type = X.weight_transfer_type.map(
        {
            "Failure Weight Transfer": 0,
            "Successful Weight Transfer": 1,
        }
    )
    X.step_type = X.step_type.map(
        {
            "Left Step": 0,
            "Right Step": 1,
        }
    )

    if "video_id" in X.columns:
        X.drop(["video_id"], axis=1, inplace=True)
    X.dropna(thresh=X.shape[1], inplace=True)
    print("Training Balance: \n", X.weight_transfer_type.value_counts())
    y = X.weight_transfer_type
    X.drop(["weight_transfer_type"], axis=1, inplace=True)

    if column_whitelist:
        X = X[column_whitelist]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_dist = {
        "n_estimators": randint(50, 500),
        "max_depth": randint(1, 20),
    }

    # Create a random forest classifier
    rf = RandomForestClassifier(class_weight="balanced")

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
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
        "feature_importances": feature_importances,
        "data_file": data_file,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "classifier": best_rf,
        "X_train": X_train,
        "X_test": X_train,
        "y_train": y_train,
        "y_test": y_test,
    }
    return test_results


if __name__ == "__main__":
    # this one is the original "avg dataset used"
    # data_file = "./data/annotated_videos/dataset_1678732901064497000.csv"

    # this one includes more pooled stats (max)
    # data_file = "./data/annotated_videos/dataset_1679002854718304000.csv"

    # this one is 45 frame window pooled
    # data_file = "./data/annotated_videos/dataset_1679015606654767000.csv"

    # this one is 25 frame window pooled
    data_file = "./data/annotated_videos/dataset_1679016147487099000.csv"

    column_whitelist = [
        "angles_max.line_5_6__line_6_7_angle_2d_degrees",
        "angles_std.line_5_6__line_25_26_angle_2d_degrees",
        "angles_avg.line_5_6__line_6_7_angle_2d_degrees",
        # "angles_avg.line_8_9__line_9_10_angle_2d_degrees",
        # "angles_max.line_5_6__line_25_26_angle_2d_degrees",
        # "angles_max.line_2_3__line_25_26_angle_2d_degrees",
        # "angles_avg.line_1_5__line_5_6_angle_2d_degrees",
        # "angles_avg.line_2_3__line_25_26_angle_2d_degrees",
        # "angles_std.line_1_5__line_5_6_angle_2d_degrees",
    ]

    results = build_random_forest(
        data_file=data_file, column_whitelist=column_whitelist
    )
    model_id = f"random-forest-{time.time_ns()}"
    model_path = "./data/trained_models"
    model_file = f"{model_id}.pickle"
    saved_model_path = f"{model_path}/{model_file}"
    notes = "This time trained on 25 frame windows where last frame is label and only on top 3 features from previous output"
    meta = json.dumps(
        {
            "type": results["type"],
            "notes": notes,
            "data_file": data_file,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "confusion_matrix": str(results["confusion_matrix"]),
            "features": dict(results["feature_importances"]),
        },
        indent=4,
    )
    if True:
        with open(f"{model_path}/{model_id}-meta.json", "w") as f:
            f.write(meta)
        with open(saved_model_path, "wb") as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        print("Saved model to pickle!")
