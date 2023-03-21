import pandas as pd
import pickle
import time
import json
import numpy as np

# Modeling
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from scipy.stats import randint

# upsampling / downsampling
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Reporting
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)


class ModelBuilder:
    """This class is to aid in setting up training data and training various models to compare performance."""

    def __init__(
        self,
    ) -> None:
        pass

    def set_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int | None = None,
        balance_off_target: bool = False,
        upsample_minority: bool = False,
        downsample_majority: bool = False,
        use_SMOTE: bool = False,
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
        if balance_off_target:
            majority_class = X_train[X_train[self.target] == 1]
            minority_class = X_train[X_train[self.target] == 0]
            if upsample_minority:
                if use_SMOTE:
                    smote = SMOTE(random_state=random_state)
                    X_train_balanced, y_train_balanced = smote.fit_resample(
                        X_train, y_train
                    )
                    X_train_balanced = pd.DataFrame(X_train_balanced)
                    y_train_balanced = pd.DataFrame(y_train_balanced)
                    X_train = X_train_balanced
                    y_train = y_train_balanced
                else:
                    minority_upsampled = resample(
                        minority_class,
                        replace=True,
                        n_samples=len(majority_class),
                        random_state=random_state,
                    )
                    X_train_balanced = pd.concat([majority_class, minority_upsampled])
                    X_train = X_train_balanced
                    y_train = X_train[self.target]
            elif downsample_majority:
                majority_downsampled = resample(
                    majority_class,
                    replace=False,
                    n_samples=len(minority_class),
                    random_state=random_state,
                )
                X_train_balanced = pd.concat([majority_downsampled, minority_class])
                X_train_balanced = X_train_balanced.sample(
                    frac=1, random_state=random_state
                ).reset_index(drop=True)
                X_train = X_train_balanced
                y_train = X_train[self.target]

        # Report training balance and drop target value
        print(
            f"Training Balance for {self.target}:\n{X_train[self.target].value_counts()}"
        )

        # drop target column
        X_train.drop([self.target], axis=1, inplace=True)
        X_test.drop([self.target], axis=1, inplace=True)

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
            naively_balance_off_target: bool
                If True, take the target var and subsample the majority class so that the dataset it balanced.
        """
        self.data_file = path
        self.target = target
        X = pd.read_csv(path, index_col=0)

        # Rewrite values based on passed map
        if bool(value_map):
            for key in value_map.keys():
                X[key] = X[key].map(value_map[key])

        # Drop empty values
        X.dropna(thresh=X.shape[1], inplace=True)

        # Preserve whitelist and drop droplist columns
        if bool(column_whitelist):
            X = X[column_whitelist]
        if bool(drop_list):
            for col in drop_list:
                if col in X.columns:
                    X.drop([col], axis=1, inplace=True)

        self.y = X[target]
        self.X = X
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
            self.feature_importances = best_rf.feature_importances_
            self.model = best_rf

            # Print the best hyperparameters
            print("Best random search hyperparameters:", rand_search.best_params_)

            # Generate predictions with the best model
            y_pred = best_rf.predict(X_test)
        else:
            rf = RandomForestClassifier(
                **params,
            )
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate model
            scores = cross_val_score(
                rf, self.X, self.y, scoring="roc_auc", cv=cv, n_jobs=-1
            )
            # summarize performance
            self.auc_cv = np.mean(scores)
            print("Mean ROC AUC from cross validation: %.3f" % self.auc_cv)
            print("Min ROC AUC from cross validation: %.3f" % min(scores))
            print("Max ROC AUC from cross validation: %.3f" % max(scores))
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            self.feature_importances = rf.feature_importances_
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
        print(classification_report(y_test, y_pred))

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
