import pandas as pd
import pickle
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from scipy.stats import randint

# upsampling / downsampling
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# feature selection
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

# Reporting
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    classification_report,
)


class ModelBuilder:
    """This class is to aid in setting up training data and training various models to compare performance."""

    def __init__(
        self,
    ) -> None:
        self.run_PCA = False

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
                    X_train_balanced = X_train_balanced.sample(
                        frac=1, random_state=random_state
                    ).reset_index(drop=True)

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
        # just a little flag for reporting
        self.run_PCA = True

        X_train = self.X_train
        X_test = self.X_test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=num_components)
        pca.fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print("PCA Variance Ratio", pca.explained_variance_ratio_)
        self.X_train = pd.DataFrame(X_train_pca)
        self.X_test = pd.DataFrame(X_test_pca)

    def evaluate_model(self):
        X_test = self.X_test
        y_test = self.y_test
        model = self.model
        # Make predictions using the trained model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(
            model, self.X, self.y, scoring="roc_auc", cv=cv, n_jobs=-1
        )
        # summarize performance
        self.auc_cv = np.mean(scores)
        print("Mean ROC AUC from cross validation: %.3f" % self.auc_cv)
        print("Min ROC AUC from cross validation: %.3f" % min(scores))
        print("Max ROC AUC from cross validation: %.3f" % max(scores))

        if not self.run_PCA:
            print("Top 5 features")
            for feature in self.feature_importances[:5]:
                print(feature)

        # Display classification metrics
        print("Classification Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        # store stats
        self.f1 = f1
        self.auc = auc
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.confusion_matrix = cm

        # Plot confusion matrix
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actual Negative", "Actual Positive"],
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.show()

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    def sort_and_format_feature_importances(self, importances):
        sorted_indices = np.argsort(importances)[::-1]

        # Get the feature names from the sorted indices
        feature_names = np.array(self.X.columns)
        sorted_features = feature_names[sorted_indices]
        output = []
        for feature, importance in zip(sorted_features, np.sort(importances)[::-1]):
            output.append({feature: f"{importance * 100}%"})
        return output

    def train_random_forest(
        self,
        use_random_search: bool = False,
        params: dict = {},
        param_dist: dict = {},
        iterations: int = 10,
        random_state: int = 123,
    ):
        """Train a random forest classifier directly or via random hyperparam search

        Args:
            use_random_search: bool
                If True, do a random search over the params dist keys/values
            params: dict
                Scikit Learn Random Forest Params to pass in
            param_dist: dict
                params for various hyperparameters to direct the random search
            iterations: int
                if we're using a random search, how many iterations should we run?

        """
        X_train = self.X_train
        y_train = self.y_train

        if use_random_search:
            rf = RandomForestClassifier(class_weight="balanced")
            rand_search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=iterations,
                cv=5,
                random_state=random_state,
            )

            # # Fit the random search object to the data
            rand_search.fit(X_train, y_train)
            # Create a variable for the best model
            best_rf = rand_search.best_estimator_
            self.feature_importances = self.sort_and_format_feature_importances(
                best_rf.feature_importances_
            )
            self.model = best_rf

            # Print the best hyperparameters
            print("Best random search hyperparameters:", rand_search.best_params_)
        else:
            rf = RandomForestClassifier(
                **params,
            )
            # evaluate model
            rf.fit(X_train, y_train)
            self.feature_importances = self.sort_and_format_feature_importances(
                rf.feature_importances_
            )
            self.model = rf

        self.model_type = "Random Forest"

        # print("Confusion matrix:")
        # print(self.confusion_matrix)
        # print("Accuracy:", self.accuracy)
        # print("Precision:", self.precision)
        # print("Recall:", self.recall)
        # print(classification_report(y_test, y_pred))

    def run_recursive_feature_estimation(self, num_features):
        if num_features:
            rfe = RFE(estimator=self.model, n_features_to_select=num_features, step=1)
            rfe.fit(self.X_train, self.y_train)
            # Get the feature rankings (1 means selected)
            feature_ranking = rfe.ranking_

            ranked_features = list(zip(self.X.columns, feature_ranking))
            sorted_ranked_features = sorted(ranked_features, key=lambda x: x[1])
            # Print the feature names and their rankings
            print("Feature Ranks (top 15)")
            for feature, rank in sorted_ranked_features[:15]:
                print(f"{feature}: {rank}")

            # Transform the training and test data using the RFE-selected features
            self.X_train = rfe.transform(self.X_train)
            self.X_test = rfe.transform(self.X_test)
            self.model.fit(self.X_train, self.y_train)
        else:
            min_features_to_select = 1  # Minimum number of features to consider
            cv = StratifiedKFold(5)

            rfecv = RFECV(
                estimator=self.model,
                step=1,
                cv=cv,
                scoring="roc_auc",
                min_features_to_select=min_features_to_select,
                n_jobs=2,
            )
            rfecv.fit(self.X_train, self.y_train)
            pass

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
                "features": self.feature_importances
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
