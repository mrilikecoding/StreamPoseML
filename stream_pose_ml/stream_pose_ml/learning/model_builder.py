import pandas as pd
import pickle
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy


# Modeling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Unsupervised
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    matthews_corrcoef,
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

    @staticmethod
    def validate_string(candidate_string: str, filters: dict[str, list[str]]) -> bool:
        """
        Determines whether a string meets multiple filter criteria.

        Usage: can be used to filter down the columns in a dataframe. i.e.
        filtered = [col for col in columns if validate_string(col, filters)]
        X = X[filtered]

        TODO move to a utility file

        Args:
            col: str
                a string to compare againt
            filters: dict
                a filter dictionary with this structure
                    {
                        "WHITELIST": [any of these substrings within tested string should return True],
                        "BLACKLIST": [any of these substrings within tested string should return False],
                        "OR": [if ANY of these substrings are within the string, return True],
                        "AND": [if ALL of these substrings are within the string, return True]
                    }

        Returns:
            True if the column name meets the filter criteria within the passed dictionary

        """
        whitelist_match = any(item in candidate_string for item in filters["WHITELIST"])
        blacklist_match = any(item in candidate_string for item in filters["BLACKLIST"])
        or_filters_match = any(filter in candidate_string for filter in filters["OR"])
        and_filters_match = all(filter in candidate_string for filter in filters["AND"])

        return (
            whitelist_match or (or_filters_match and and_filters_match)
        ) and not blacklist_match

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
        auc = roc_auc_score(y_test, y_pred_proba, average="weighted")

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(
            model, self.X, self.y, scoring="roc_auc", cv=cv, n_jobs=-1
        )
        # summarize performance
        self.auc_cv = np.mean(scores)
        print("Mean ROC AUC from cross validation: %.3f" % self.auc_cv)
        print("Min ROC AUC from cross validation: %.3f" % min(scores))
        print("Max ROC AUC from cross validation: %.3f" % max(scores))

        if not self.run_PCA and hasattr(self, "feature_importances"):
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
        # Calculate the Matthews correlation coefficient
        mcc = matthews_corrcoef(y_test, y_pred)
        print(f"Matthews correlation coefficient (-1 to 1): {mcc}")

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

    def train_gradient_boost(self):
        X_train = self.X_train
        y_train = self.y_train
        gradient_booster = xgb.XGBClassifier(eval_metric="mlogloss")
        gradient_booster.fit(X_train, y_train)
        self.model = gradient_booster

    def train_logistic_regression(self):
        X_train = self.X_train
        y_train = self.y_train
        logisitic_regression = LogisticRegression(
            solver="saga", max_iter=1000, random_state=42
        )
        # logisitic_regression.fit(X_train, y_train)
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("logreg", logisitic_regression)]
        )
        param_grid = {
            "logreg__tol": [1e-4, 1e-5, 1e-6],
            "logreg__C": [0.1, 1, 10],
            "logreg__penalty": ["l1", "l2"],
        }
        grid_search = GridSearchCV(
            pipeline, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

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
            random_state: int
                if we're using a random search, pass in a random see to keep the same randomness across runs

        """
        X_train = self.X_train
        y_train = self.y_train

        if use_random_search:
            rf = RandomForestClassifier()
            rand_search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=iterations,
                cv=5,
                random_state=random_state,
                scoring="roc_auc",
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

    def save_model_and_datasets(self, notes: str, model_type: str = None, model_path: str = "../../data/trained_models"):
        """Save the current model and metadata to a pickle / json file.

        Args:
            notes: str
                notes to explain things about this particular dataset/model
        """
        if hasattr(self, "model_type"):
            model_type = self.model_type

        model_id = f"{model_type}-{time.time_ns()}"
        model_file = f"{model_id}.pickle"
        saved_model_path = f"{model_path}/{model_file}"
        meta = json.dumps(
            {
                "type": model_type,
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
            "type": model_type,
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
            "columns": self.X_test.columns.tolist(),
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }
        with open(f"{model_path}/{model_id}-meta.json", "w") as f:
            f.write(meta)
        with open(saved_model_path, "wb") as f:
            pickle.dump(model_data, f, pickle.HIGHEST_PROTOCOL)

        print(f"Saved model to pickle! {saved_model_path}")

    def retrieve_model_from_pickle(self, file_path: str):
        """Load a model and metadata from a pickle file.

        Args:
            file_path: str
                The location of the pickle file to load.

        Returns:
            tuple: model, model_data
                The loaded model and associated model data in the pickle
        """
        with open(file_path, "rb") as f:
            model_data = pickle.load(f)

        # Set the model builder class attributes from the loaded model data
        self.model_type = model_data["type"]
        self.feature_importances = model_data["feature_importances"]
        self.data_file = model_data["data_file"]
        self.auc = model_data["auc-roc"]
        self.accuracy = model_data["accuracy"]
        self.precision = model_data["precision"]
        self.recall = model_data["recall"]
        self.confusion_matrix = model_data["confusion_matrix"]
        self.model = model_data["classifier"]
        self.columns = model_data["columns"]
        self.X_train = model_data["X_train"]
        self.X_test = model_data["X_test"]
        self.y_train = model_data["y_train"]
        self.y_test = model_data["y_test"]

        print("Loaded model from pickle!")

        return self.model, model_data

    def find_k_means_clusters(
        self,
        X: pd.DataFrame = None,
        n_clusters: int = 3,
        random_state: int = None,
        cluster_range: tuple[int, int] = None,
    ) -> KMeans:
        """Trains a Kmeans algorithm based on passed number of clusters.
        Or finds optimal number of clusters based on passed range

            Args:
                n_clusters: int
                    How many clusters in the kmeans
                random_state: int
                    preserve state
                cluster_range: tuple[int, int]
                    a start and end number for the range of cluster numbers to try
            Returns:
                the KMeans classifer fit to the data
        """

        if X is None:
            X = self.X

        if cluster_range is not None:
            # within cluster sum of squares - smaller is better
            # how close are the points within the cluster?
            wcss = {}
            for n in range(cluster_range[0], cluster_range[1]):
                kmeans = KMeans(n_init="auto", n_clusters=n, random_state=random_state)
                kmeans.fit(X)
                wcss[n] = kmeans.inertia_
            plt.plot(
                range(cluster_range[0], cluster_range[1]), wcss.values(), marker="o"
            )
            plt.xlabel("Number of Clusters")
            plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
            plt.title("Elbow Curve")
            plt.grid()
            plt.show()
            knee_locator = KneeLocator(
                range(cluster_range[0], cluster_range[1]),
                list(wcss.values()),
                curve="convex",
                direction="decreasing",
            )
            optimal_clusters = knee_locator.knee
            print("Optimal Number of Clusters:", optimal_clusters)
            kmeans_optimal = KMeans(
                n_init="auto", n_clusters=optimal_clusters, random_state=random_state
            )
            kmeans_optimal.fit(X)
            print("Optimal K-means Scores:")
            self.k_means_metrics(X=X, kmeans=kmeans_optimal)
            return kmeans_optimal
        else:
            kmeans = KMeans(X=X, n_clusters=n_clusters, random_state=random_state)
            kmeans.fit(X)
            self.k_means_metrics(
                X=X,
                kmeans=kmeans,
            )
            return kmeans

    def k_means_metrics(self, kmeans: KMeans, X: pd.DataFrame = None) -> None:
        print("K Means Evaluation")
        if X is None:
            X = self.X
        # Silhouette Score
        sil_score = silhouette_score(X, kmeans.labels_)
        print("Silhouette Score:", sil_score)

        # Between Clusters Sum of Squares (BCSS)
        WCSS = kmeans.inertia_
        total_sum_of_squares = np.sum(np.var(X, axis=0) * (X.shape[0] - 1))
        BCSS = total_sum_of_squares - WCSS
        print("")
        print("Between Clusters Sum of Squares (BCSS):", BCSS)

        # Sum of Squares Error (SSE)
        print("")
        print("Sum of Squares Error (SSE):", WCSS)

        # Maximum Radius
        clusters = kmeans.cluster_centers_
        distances = np.linalg.norm(X - clusters[kmeans.labels_], axis=1)
        max_radius = np.max(distances)
        print("")
        print("Maximum Radius:", max_radius)

        # Average Radius
        avg_radius = np.sum(distances) / X.shape[0]
        print("")
        print("Average Radius:", avg_radius)

        # Assuming kmeans is your trained KMeans model
        labels = kmeans.labels_
        # Count the occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Create a bar plot
        plt.bar(unique_labels, counts)

        # Set x-axis labels and a title for the plot
        plt.xticks(unique_labels, [f"Cluster {label}" for label in unique_labels])
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.title("Distribution of K-means Labels")

        # Show the plot
        plt.show()

    @staticmethod
    def get_cluster_subset(
        kmeans: KMeans, X: pd.DataFrame, cluster_list: list
    ) -> pd.DataFrame:
        """
        Get the subset of data corresponding to the specified clusters.

        Args:
            kmeans (KMeans): A trained KMeans classifier.
            X (pd.DataFrame): The original feature matrix.
            cluster_list (list): A list of cluster numbers you want to include in the subset.

        Returns:
            pd.DataFrame: A DataFrame containing the subset of data for the specified clusters.
        """
        # Get the cluster assignments
        cluster_assignments = kmeans.labels_

        # Combine the features and cluster assignments into a single DataFrame
        data_with_clusters = X.copy()
        data_with_clusters["cluster"] = cluster_assignments

        # Get the subset of data for the specified clusters
        cluster_subset = data_with_clusters[
            data_with_clusters["cluster"].isin(cluster_list)
        ]

        # Drop the "cluster" column to return only the original features
        cluster_subset = cluster_subset.drop("cluster", axis=1)

        return cluster_subset
