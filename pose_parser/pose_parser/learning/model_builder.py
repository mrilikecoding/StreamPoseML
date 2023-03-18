import pandas as pd
import pickle
import time
import json

# Modeling
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Reporting
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelBuilder:
    """This class is to aid in setting up training data and training various models to compare performance."""

    def __init__(self) -> None:
        pass

    def set_train_test_split(
        self, test_size: float = 0.2, random_state: int | None = None
    ) -> None:
        """Set the train test split based on test size (80/20 by default)

        Args:
            test_size: float
                Percentage to use for testing.
            random_state: int | None
                Pass this value consistently to get the same train/test split

        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def load_and_prep_dataset_from_csv(
        self,
        path: str,
        target: str,
        value_map: dict = {},
        drop_list: list = [],
        column_whitelist: list = [],
    ):
        """Load CSV into Pandas dataframe with no index column

        Args:
            path: str
                location of CSV dataset
            target: str
                the target variable we want to predict
            value_map: dict
                if there are string values in the columns map the categorical values to numbers
                according to the passed value map
            column_whitelist: list
                only keep these columns in the dataset
        """
        self.data_file = path
        self.X = pd.read_csv(path, index_col=0)

        # Rewrite values based on passed map
        if bool(value_map):
            for key in value_map.keys():
                self.X[key] = self.X[key].map(value_map[key])

        # Drop empty values
        self.X.dropna(thresh=self.X.shape[1], inplace=True)

        # Report training balance and drop target value
        print(f"Training Balance for {target}:\n{self.X[target].value_counts()}")
        self.y = self.X[target]
        self.X.drop([target], axis=1, inplace=True)

        # Preserve whitelist and drop droplist columns
        if bool(column_whitelist):
            self.X = self.X[column_whitelist]
        if bool(drop_list):
            for col in drop_list:
                if col in self.X.columns:
                    self.X.drop([col], axis=1, inplace=True)

        return True

    def run_pca(self, num_components: int = 5):
        """Use PCA (Principle Component Analysis) to transform the
        dataset into a certain number of components

        Args:
            num_components: int
                number of components to use
        """
        pca = PCA(n_components=num_components)
        X_pca = pca.fit_transform(self.X)
        X_pca_inv = pca.inverse_transform(X_pca)
        print("PCA Variance Ratio", pca.explained_variance_ratio_)
        self.X = pd.DataFrame(X_pca_inv)

    def train_random_forest(
        self, use_random_search: bool = False, params: dict = {}, param_dist: dict = {}
    ):
        """Train a random forest classifier directly or via random hyperparam search

        Args:
            use_random_search: bool
                If True, do a random search over the params dist keys/values
            params: dict
                Scikit Learn Random Forest Params to pass in
            param_dist: dict
                params for various hyperparameters to direct the random search

        """
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        y_pred = None

        if use_random_search:
            rf = RandomForestClassifier(class_weight="balanced")
            rand_search = RandomizedSearchCV(
                rf, param_distributions=param_dist, n_iter=10, cv=5
            )

            # # Fit the random search object to the data
            rand_search.fit(X_train, y_train)
            # Create a variable for the best model
            best_rf = rand_search.best_estimator_
            self.model = best_rf

            # Print the best hyperparameters
            print("Best random search hyperparameters:", rand_search.best_params_)

            # Generate predictions with the best model
            y_pred = best_rf.predict(X_test)
        else:
            rf = RandomForestClassifier(
                **params,
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            self.model = rf

        self.model_type = "Random Forest"

        self.auc = roc_auc_score(y_test, y_pred)
        # Create the confusion matrix
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred)
        self.recall = recall_score(y_test, y_pred)

        print("Confusion matrix:")
        print(self.confusion_matrix)
        print("Accuracy:", self.accuracy)
        print("Precision:", self.precision)
        print("Recall:", self.recall)

    def save_model_and_datasets(self, notes: str):
        """Save the current model and metadata to a pickle / json file.

        Args:
            notes: str
                notes to explain things about this particular dataset/model
        """
        model_id = f"{self.model_type}-{time.time_ns()}"
        model_path = "./data/trained_models"
        model_file = f"{model_id}.pickle"
        saved_model_path = f"{model_path}/{model_file}"
        meta = json.dumps(
            {
                "type": self.model_type,
                "notes": notes,
                "data_file": self.data_file,
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "confusion_matrix": str(self.confusion_matrix),
                "features": dict(self.feature_importances)
                if hasattr(self, "feature_importances")
                else None,
            },
            indent=4,
        )
        model_data = {
            "type": self.model_type,
            "feature_importances": self.feature_importances
            if hasattr(self, "feature_importances")
            else None,
            "data_file": self.data_file,
            "auc-roc": self.auc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "confusion_matrix": self.confusion_matrix,
            "classifier": self.model,
            "X_train": self.X_train,
            "X_test": self.X_train,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }
        with open(f"{model_path}/{model_id}-meta.json", "w") as f:
            f.write(meta)
        with open(saved_model_path, "wb") as f:
            pickle.dump(model_data, f, pickle.HIGHEST_PROTOCOL)

        print("Saved model to pickle!")

    def report(self):
        """Print model details."""
        print("Type", self.model_type)
        print("Data_file", self.data_file)
        print("AUC", self.auc)
        print("Accuracy", self.accuracy)
        print("Precision", self.precision)
        print("Recall", self.recall)
        print("Confusion_matrix", self.confusion_matrix)
        print(
            "Feature_importances",
            self.feature_importances if hasattr(self, "feature_importances") else None,
        )


if __name__ == "__main__":
    # TODO move this exploration into a Jupyter Notebook

    # this one is the original "avg dataset used"
    # data_file = "./data/annotated_videos/dataset_1678732901064497000.csv"

    # this one includes more pooled stats (max)
    data_file = "./data/annotated_videos/dataset_1679002854718304000.csv"

    # this one is 45 frame window pooled
    # data_file = "./data/annotated_videos/dataset_1679015606654767000.csv"

    # this one is 25 frame window pooled
    # data_file = "./data/annotated_videos/dataset_1679016147487099000.csv"

    # # this one is a flat column representation frame by frame angles of a labeled 10 frame window
    # data_file = "./data/annotated_videos/dataset_1679087888313443000.csv"

    value_map = {
        "weight_transfer_type": {
            "Failure Weight Transfer": 0,
            "Successful Weight Transfer": 1,
        },
        "step_type": {
            "Left Step": 0,
            "Right Step": 1,
        },
    }
    drop_list = ["video_id"]
    column_whitelist = [
        # "angles_max.line_5_6__line_6_7_angle_2d_degrees",
        # "angles_std.line_5_6__line_25_26_angle_2d_degrees",
        # "angles_avg.line_5_6__line_6_7_angle_2d_degrees",
        # "angles_avg.line_8_9__line_9_10_angle_2d_degrees",
        # "angles_max.line_5_6__line_25_26_angle_2d_degrees",
        # "angles_max.line_2_3__line_25_26_angle_2d_degrees",
        # "angles_avg.line_1_5__line_5_6_angle_2d_degrees",
        # "angles_avg.line_2_3__line_25_26_angle_2d_degrees",
        # "angles_std.line_1_5__line_5_6_angle_2d_degrees",
    ]

    """ WRITE NOTES ON THIS RUN HERE """
    notes = """
    Dataset notes:
    Flat column representation of 10 windows of frame data angles

    Model notes:
    PCA on a rand search Random Forest. 
    """

    mb = ModelBuilder()
    mb.load_and_prep_dataset_from_csv(
        path=data_file,
        target="weight_transfer_type",
        value_map=value_map,
        column_whitelist=column_whitelist,
        drop_list=drop_list,
    )
    # mb.run_pca(num_components=5)
    mb.set_train_test_split(random_state=123)

    param_dist = {
        "n_estimators": randint(50, 500),
        "max_depth": randint(1, 20),
        "max_features": randint(3, 20),
    }
    rf_params = {"class_weight": "balanced", "n_estimators": 500, "max_depth": 25}

    mb.train_random_forest(
        use_random_search=False, params=rf_params, param_dist=param_dist
    )
    mb.report()
    if False:
        mb.save_model_and_datasets(notes=notes)
