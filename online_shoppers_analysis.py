'''
Annabel Luo, Elle Zhou
ACAD 191, Fall 2026
luoannab@usc.edu
Exercise 8
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas.plotting import scatter_matrix
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_validate, RandomizedSearchCV, GridSearchCV,
                                     FixedThresholdClassifier)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_FOLDER = Path(__file__).resolve().parent
PURCHASE_STYLES = [
    (False, "#0072B2", "No purchase (Revenue=False)"),
    (True, "#D55E00", "Purchase (Revenue=True)"),
]


# Function: load_data
# Purpose: Read a local CSV and handle possible loading errors.
# Parameters: filename - name of the CSV file
# Side effects: Prints an error message if loading fails
# Returns: DataFrame if successful, None otherwise
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print("Error: CSV file not found:", filename)
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: Could not read the CSV format.")
    except OSError as error:
        print("Error opening file:", error)
    return None


# Function: explore_data
# Purpose: Print dataset information, summary statistics, and category counts.
# Parameters: data - the dataset as a DataFrame
# Side effects: Prints information to the console
# Returns: None
def explore_data(data):
    print("\nDATASET SIZE")
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    print("\nDATASET INFORMATION (info())")
    data.info()

    print("\nNUMERICAL SUMMARY (describe())")
    print(data.describe().to_string())

    print("\nVISITOR TYPES (value_counts())")
    print(data["VisitorType"].value_counts().to_string())

    print("\nPURCHASE OUTCOMES (value_counts())")
    print(data["Revenue"].value_counts().to_string())

    print("\nMISSING VALUES")
    print(data.isnull().sum().sum())


# Function: visualize_data
# Purpose: Create histograms for the numerical dataset columns.
# Parameters: data - the dataset as a DataFrame
# Side effects: Creates and saves histograms
# Returns: None
def visualize_data(data):
    data.hist(figsize=(14, 12), bins=20)
    plt.suptitle("Online Shoppers - Numerical Attribute Histograms")
    plt.tight_layout()
    plt.savefig(PROJECT_FOLDER / "online_shoppers_histograms.png", dpi=150)


# Function: scatter_plot
# Purpose: Compare product browsing behavior for buyers and non-buyers.
# Parameters: data - the dataset as a DataFrame
# Side effects: Creates and saves a scatter plot
# Returns: None
def scatter_plot(data):
    figure, axis = plt.subplots(figsize=(10, 7))
    for purchased, color, label in PURCHASE_STYLES:
        group = data[data["Revenue"] == purchased]
        axis.scatter(group["ProductRelated"], group["ProductRelated_Duration"],
                     color=color, label=label, alpha=0.35, s=15)
    axis.set_title("Product Browsing Behavior by Purchase Outcome")
    axis.set_xlabel("Number of Product-Related Pages Visited")
    axis.set_ylabel("Total Time on Product-Related Pages")
    axis.legend(title="Purchase outcome", markerscale=2)
    figure.tight_layout()
    figure.savefig(PROJECT_FOLDER / "online_shoppers_scatter.png", dpi=150)


# Function: scatter_all_numeric
# Purpose: Compare every pair of numeric columns, colored by purchase outcome.
# Parameters: data - the dataset as a DataFrame
# Side effects: Creates and saves a scatter plot matrix
# Returns: None
def scatter_all_numeric(data):
    # Draw buyers last so the less common purchase class remains visible.
    plot_data = data.sort_values("Revenue", kind="stable")
    numeric_data = plot_data.select_dtypes(include="number")
    color_map = {purchased: color for purchased, color, label in PURCHASE_STYLES}
    colors = plot_data["Revenue"].map(color_map).to_numpy()
    print("\nSCATTER MATRIX - ALL NUMERIC COLUMNS")
    print(numeric_data.columns.to_list())
    print("Rows plotted:", len(numeric_data))

    axes = scatter_matrix(numeric_data, figsize=(28, 28), alpha=0.3,
                          c=colors, s=4, diagonal="hist",
                          hist_kwds={"bins": 20, "color": "#8c8c8c"})
    for row in axes:
        for axis in row:
            axis.tick_params(axis="both", labelsize=6)
            axis.set_xlabel(axis.get_xlabel(), fontsize=8, rotation=90)
            axis.set_ylabel(axis.get_ylabel(), fontsize=8)
    figure = axes[0, 0].figure
    figure.suptitle("Online Shoppers - All Numeric Columns by Purchase Outcome",
                   fontsize=18, y=0.997)
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for purchased, color, label in PURCHASE_STYLES
    ]
    figure.legend(handles=legend_handles, loc="upper center", ncol=2,
                  fontsize=12, bbox_to_anchor=(0.5, 0.985))
    figure.text(0.5, 0.966, "Diagonal: histograms of all visits combined",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(PROJECT_FOLDER / "online_shoppers_scatter_matrix.png", dpi=150)


# Function: correlations
# Purpose: Calculate Pearson linear correlations for numeric attributes.
# Parameters: data - the dataset as a DataFrame
# Side effects: Prints the correlation matrix and saves it as a CSV
# Returns: None
def correlations(data):
    # Match the example: include every numeric column and Boolean column.
    # OS, browser, region, and traffic codes are nominal categories; their
    # coefficients depend on arbitrary numbering and need careful interpretation.
    corr_matrix = data.corr(numeric_only=True)

    print("\nLINEAR CORRELATIONS (corr())")
    print(corr_matrix.round(3).to_string())
    print("Note: OperatingSystems, Browser, Region, and TrafficType are category codes.")

    print("\nCORRELATIONS WITH PURCHASE OUTCOME (Revenue)")
    print(corr_matrix["Revenue"].drop("Revenue")
          .sort_values(ascending=False).round(3).to_string())

    corr_matrix.to_csv(PROJECT_FOLDER / "online_shoppers_correlations.csv")


# Custom transformer: remove selected attributes and add a browsing measure.
class ShoppingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, excluded_features=("Informational_Duration", "Weekend", "SpecialDay")):
        self.excluded_features = excluded_features

    def fit(self, X, y=None):
        # These changes use each row's values, so no statistics are learned.
        return self

    def transform(self, X):
        prepared = X.drop(columns=list(self.excluded_features)).copy()
        prepared["ProductTimePerPage"] = 0.0
        has_product_pages = prepared["ProductRelated"] > 0
        prepared.loc[has_product_pages, "ProductTimePerPage"] = (
            prepared.loc[has_product_pages, "ProductRelated_Duration"]
            / prepared.loc[has_product_pages, "ProductRelated"]
        )
        return prepared


# Function: prepare_data
# Purpose: Split data and fit a reusable feature preparation pipeline.
# Parameters: data - the original dataset as a DataFrame
# Side effects: Prints the prepared feature sizes and an encoding preview
# Returns: Original train/test features, train/test labels, and fitted pipeline
def prepare_data(data):
    excluded_features = ("Informational_Duration", "Weekend", "SpecialDay")
    X = data.drop(columns=["Revenue"])
    y = data["Revenue"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # The integer codes are categories too; their numbers imply no ranking.
    categorical_columns = ["Month", "VisitorType", "OperatingSystems",
                           "Browser", "Region", "TrafficType"]
    numeric_columns = ["Administrative", "Administrative_Duration", "Informational",
                       "ProductRelated", "ProductRelated_Duration", "BounceRates",
                       "ExitRates", "PageValues", "ProductTimePerPage"]

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    column_preparation = ColumnTransformer(
        [("numeric", numeric_pipeline, numeric_columns),
         ("categorical", categorical_pipeline, categorical_columns)],
        remainder="drop",
        verbose_feature_names_out=False
    )
    # No imputer is needed: the current dataset has no missing values.
    preprocessor = Pipeline([
        ("feature_preparation", ShoppingFeatures(excluded_features=excluded_features)),
        ("column_preparation", column_preparation)
    ])

    # Learn scaling and categories from training data only, then reuse for test.
    train_values = preprocessor.fit_transform(X_train)
    test_values = preprocessor.transform(X_test)
    feature_names = preprocessor.named_steps["column_preparation"].get_feature_names_out()
    X_train_prepared = pd.DataFrame(train_values, columns=feature_names,
                                    index=X_train.index).astype(float)
    X_test_prepared = pd.DataFrame(test_values, columns=feature_names,
                                   index=X_test.index).astype(float)

    print("\nDATA PREPARATION PIPELINE")
    print("Attributes excluded from model inputs:", excluded_features)
    print("New attribute: ProductTimePerPage = product browsing time / product pages")
    print("ProductTimePerPage is 0 when no product pages were visited.")
    print("Standardized numeric columns:", numeric_columns)
    print("Encoded categorical columns:", categorical_columns)
    print(preprocessor)
    print("Training features:", X_train_prepared.shape)
    print("Test features:", X_test_prepared.shape)
    print("Revenue labels: 0 = no purchase; 1 = purchase")
    print("Preview of encoded Month and VisitorType columns:")
    preview_columns = [name for name in feature_names
                       if name.startswith(("Month_", "VisitorType_"))]
    print(X_train_prepared[preview_columns].head().to_string())

    # Return original columns so cross-validation can refit preprocessing per fold.
    return X_train, X_test, y_train, y_test, preprocessor


# Function: compare_models
# Purpose: Train two classifiers and compare five-fold cross-validation F1 scores.
# Parameters: X_train, y_train - training data; preprocessor - preparation pipeline
# Side effects: Prints predictions, training F1, fold scores, mean, and deviation
# Returns: A dictionary containing each model's mean cross-validation F1
def compare_models(X_train, y_train, preprocessor):
    forest_clf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        bootstrap=True,
        oob_score=False,
        random_state=42
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": forest_clf
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\nMODEL COMPARISON - 5-FOLD CROSS-VALIDATION")
    print("Score: F1 for purchase (Revenue=1). Higher is better.")
    print("Only the training set is used; the separate test set is reserved.")
    for name, model in models.items():
        model_pipeline = Pipeline([
            ("preparation", clone(preprocessor)),
            ("classifier", model)
        ])
        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_train)
        print("\n" + name)
        print("Predictions:", predictions[:5])
        print("Labels:", y_train.iloc[:5].to_list())
        print("Training F1:", round(f1_score(y_train, predictions), 4))

        # Include preprocessing inside CV so each fold learns its own settings.
        scores = cross_val_score(model_pipeline, X_train, y_train,
                                 scoring="f1", cv=cv, error_score="raise")
        print("CV Scores:", scores.round(4))
        print("CV Mean:", round(scores.mean(), 4))
        print("CV Standard deviation:", round(scores.std(), 4))
        results[name] = scores.mean()



# training and testing - round 1
# Reference: load_data6.py, adapted to classification and RandomizedSearchCV.
# Test each sampled setting on validation folds; reserve the final test set.
def tune_random_forest(X_train, y_train, preprocessor):
    forest_clf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        bootstrap=True,
        oob_score=False,
        random_state=42
    )
    forest_pipeline = Pipeline([
        ("preparation", clone(preprocessor)),
        ("classifier", forest_clf)
    ])
    # Search overrides these two starting values; the other settings stay fixed.
    param_distributions = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_features": ["sqrt", "log2", 0.3]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        forest_pipeline, param_distributions, n_iter=5, cv=cv,
        scoring="f1", random_state=42, refit=True,
        return_train_score=True, error_score="raise"
    )

    print("\nTRAINING AND TESTING - ROUND 1: RANDOMIZED SEARCH")
    print("5 sampled combinations x 5 folds = 25 CV fits, then refit the best.")
    print("Score: F1 for purchase (Revenue=1). Higher is better.")
    random_search.fit(X_train, y_train)

    # Like load_data6.py, print the score for each tested parameter combination.
    cvres = random_search.cv_results_
    for mean_score, std_score, params in zip(
            cvres["mean_test_score"], cvres["std_test_score"], cvres["params"]):
        print("CV F1:", round(mean_score, 4),
              "Standard deviation:", round(std_score, 4), params)

    print("Best parameters:", random_search.best_params_)
    print("Best mean CV F1:", round(random_search.best_score_, 4))
    print("Final test set remains reserved until tuning is finished.")
    return random_search


# training and testing - round 2
# Reference: load_data6.py; refine round 1 using GridSearchCV.
# More features may help splits; larger leaves may reduce overfitting.
def tune_random_forest_round2(X_train, y_train, preprocessor):
    forest_clf = RandomForestClassifier(
        n_estimators=300,
        max_features=0.3,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        oob_score=False,
        random_state=42
    )
    forest_pipeline = Pipeline([
        ("preparation", clone(preprocessor)),
        ("classifier", forest_clf)
    ])
    param_grid = {
        "classifier__n_estimators": [300, 500],
        "classifier__max_features": [0.3, 0.4, 0.5],
        "classifier__min_samples_leaf": [1, 2, 4]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        forest_pipeline, param_grid, cv=cv,
        scoring={"f1": "f1", "precision": "precision", "recall": "recall"},
        refit="f1", n_jobs=2,
        return_train_score=True, error_score="raise", verbose=1
    )

    print("\nTRAINING AND TESTING - ROUND 2: GRID SEARCH")
    print("18 combinations x 5 folds = 90 CV fits, then refit the best.")
    print("Use the same split, preprocessing and folds as round 1.")
    grid_search.fit(X_train, y_train)

    cvres = grid_search.cv_results_
    round1_index = None
    for index, params in enumerate(cvres["params"]):
        print("CV F1:", round(cvres["mean_test_f1"][index], 4),
              "Standard deviation:", round(cvres["std_test_f1"][index], 4),
              "Precision:", round(cvres["mean_test_precision"][index], 4),
              "Recall:", round(cvres["mean_test_recall"][index], 4), params)
        if (params["classifier__n_estimators"] == 300
                and params["classifier__max_features"] == 0.3
                and params["classifier__min_samples_leaf"] == 1):
            round1_index = index

    print("Best parameters:", grid_search.best_params_)
    print("Best mean CV F1:", round(grid_search.best_score_, 4))
    best = grid_search.best_index_
    print("Best model mean training F1:", round(cvres["mean_train_f1"][best], 4))
    if round1_index is not None:
        baseline_f1 = cvres["mean_test_f1"][round1_index]
        print("Round 1 winner rechecked on the same folds:", round(baseline_f1, 4))
        print("F1 change from round 1:", round(grid_search.best_score_ - baseline_f1, 4))
    print("Final test set remains reserved; these are cross-validation results.")
    return grid_search


# training and testing - round 3
# Reference: load_data6.py; use grid search to target missed purchasers.
# Choose the threshold inside CV along with forest settings, never on the test set.
def tune_random_forest_round3(X_train, y_train, preprocessor):
    forest_clf = RandomForestClassifier(
        n_estimators=500,
        max_features=0.5,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=4,
        bootstrap=True,
        oob_score=False,
        random_state=42
    )
    forest_pipeline = Pipeline([
        ("preparation", clone(preprocessor)),
        ("classifier", forest_clf)
    ])
    threshold_clf = FixedThresholdClassifier(
        clone(forest_clf), response_method="predict_proba", pos_label=1
    )
    param_grid = [
        # Keep round 2's exact model as a control, using its original predict().
        {"classifier": [clone(forest_clf)]},
        {
            "classifier": [threshold_clf],
            "classifier__estimator__max_features": [0.5, 0.75],
            "classifier__estimator__min_samples_leaf": [4, 8],
            "classifier__estimator__class_weight": [None, "balanced"],
            "classifier__threshold": [0.25, 0.35, 0.5]
        }
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        forest_pipeline, param_grid, cv=cv,
        scoring={"f1": "f1", "precision": "precision", "recall": "recall"},
        refit="f1", n_jobs=4, return_train_score=True,
        error_score="raise", verbose=2
    )

    print("\nTRAINING AND TESTING - ROUND 3: GRID SEARCH")
    print("24 new combinations + round 2 control = 25 candidates, 125 CV fits.")
    print("Same five folds and preprocessing; choose the highest mean F1.")
    grid_search.fit(X_train, y_train)

    cvres = grid_search.cv_results_
    for index, params in enumerate(cvres["params"]):
        print("CV F1:", round(cvres["mean_test_f1"][index], 4),
              "Precision:", round(cvres["mean_test_precision"][index], 4),
              "Recall:", round(cvres["mean_test_recall"][index], 4), params)
    best = grid_search.best_index_
    print("Best parameters:", grid_search.best_params_)
    print("Best mean CV F1:", round(grid_search.best_score_, 4))
    print("Best mean precision:", round(cvres["mean_test_precision"][best], 4))
    print("Best mean recall:", round(cvres["mean_test_recall"][best], 4))
    print("Round 2 control CV F1:", round(cvres["mean_test_f1"][0], 4))
    print("Final test set remains reserved; these are cross-validation results.")
    return grid_search


# training and testing - cross-validation of the selected model
# Recheck the fixed winning settings and record all four comparison metrics.
def cross_validate_selected_model(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        clone(model), X_train, y_train, cv=cv,
        scoring={"precision": "precision", "recall": "recall",
                 "accuracy": "accuracy", "f1": "f1"},
        return_train_score=True, n_jobs=4, error_score="raise"
    )
    print("\nFINAL MODEL - FIVE-FOLD CROSS-VALIDATION")
    for metric in ("precision", "recall", "accuracy", "f1"):
        scores = results["test_" + metric]
        print(metric, "fold scores:", scores.round(4))
        print(metric, "CV mean:", round(scores.mean(), 4),
              "standard deviation:", round(scores.std(), 4))
    return results


# training and testing - final evaluation
# Run after model selection is finished. Predict with the fitted pipeline only.
def evaluate_final_model(model, X_train, y_train, X_test, y_test, cv_results=None):
    print("\nFINAL EVALUATION - SELECTED ROUND 3 MODEL")
    scores = {}
    for name, X, y in [("Training", X_train, y_train), ("Test", X_test, y_test)]:
        predictions = model.predict(X)
        scores[name] = {
            "f1": f1_score(y, predictions),
            "precision": precision_score(y, predictions),
            "recall": recall_score(y, predictions),
            "accuracy": accuracy_score(y, predictions)
        }
        print(name, "rows:", len(y))
        for metric, value in scores[name].items():
            print(name, metric + ":", round(value, 4))
        if name == "Test":
            test_predictions = predictions

    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    print("\nTEST CLASSIFICATION REPORT")
    print(classification_report(y_test, test_predictions,
                                target_names=["No purchase", "Purchase"], digits=4))
    print("TEST CONFUSION MATRIX - rows actual, columns predicted; order 0, 1")
    print(matrix)
    print("No-purchase baseline test accuracy:", round((y_test == 0).mean(), 4))
    if cv_results is not None:
        print("\nCROSS-VALIDATION VERSUS FINAL TEST SET")
        for metric in ("precision", "recall", "accuracy", "f1"):
            cv_mean = cv_results["test_" + metric].mean()
            test_score = scores["Test"][metric]
            print(metric, "CV:", round(cv_mean, 4), "Test:", round(test_score, 4),
                  "Change:", round(test_score - cv_mean, 4))
    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(matrix, display_labels=["No purchase", "Purchase"])
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.grid(False)
    axis.set_title("Selected Random Forest - Final Test Set")
    figure.tight_layout()
    figure.savefig(PROJECT_FOLDER / "online_shoppers_test_confusion_matrix.png", dpi=180)
    scores["confusion_matrix"] = matrix.tolist()
    scores["classification_report"] = classification_report(
        y_test, test_predictions, target_names=["No purchase", "Purchase"], output_dict=True
    )
    return scores


# Function: main
# Purpose: Explore the CSV, prepare features, and compare Revenue classifiers.
# Parameters: None
# Side effects: Prints dataset information and displays histograms
# Returns: None
def main():
    filename = PROJECT_FOLDER / "online_shoppers_intention.csv"
    print("\nRUNNING SCRIPT:", Path(__file__).resolve())
    print("DATA SOURCE:", filename)
    print("SCATTER COLORS: blue = no purchase; orange = purchase")
    data = load_data(filename)

    if data is not None:
        explore_data(data)
        correlations(data)
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(data)
        compare_models(X_train, y_train, preprocessor)
        # training and testing - round 1
        tune_random_forest(X_train, y_train, preprocessor)
        # training and testing - round 2
        tune_random_forest_round2(X_train, y_train, preprocessor)
        # training and testing - round 3
        round3_search = tune_random_forest_round3(X_train, y_train, preprocessor)
        # training and testing - final evaluation
        final_cv = cross_validate_selected_model(round3_search.best_estimator_, X_train, y_train)
        evaluate_final_model(round3_search.best_estimator_, X_train, y_train, X_test, y_test,
                             cv_results=final_cv)
        visualize_data(data)
        scatter_plot(data)
        scatter_all_numeric(data)
        print("\nAnalysis and model comparison completed. Charts and correlations saved in:")
        print(PROJECT_FOLDER)
        print("Close the plot windows to finish the program.")
        plt.show()


if __name__ == "__main__":
    main()
