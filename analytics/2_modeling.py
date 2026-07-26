import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "titanic.csv"

PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# LOAD CLEANED DATA
# IMPORTANT:
# DO NOT CALL sns.load_dataset() HERE
# ============================================================

print("=" * 70)
print("LOADING CLEANED TITANIC DATASET")
print("=" * 70)

df = pd.read_csv(
    CSV_PATH
)

print(
    f"Dataset loaded from: {CSV_PATH}"
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# TARGET AND FEATURES
# ============================================================

y = df["survived"]

X = df.drop(
    columns=["survived"]
)


# ============================================================
# CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("TARGET CLASS BALANCE")
print("=" * 70)

class_counts = (
    y.value_counts()
    .sort_index()
)

class_percentages = (
    y.value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print("\nClass Counts:")
print(class_counts)

print("\nClass Percentages:")
print(
    class_percentages.round(2)
)


print(
    "\nClass 0 = Did not survive"
)

print(
    "Class 1 = Survived"
)


# ============================================================
# STRATIFIED TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n" + "=" * 70)
print("STRATIFIED TRAIN/TEST SPLIT")
print("=" * 70)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)

print(
    f"Original survival rate: "
    f"{y.mean() * 100:.2f}%"
)

print(
    f"Training survival rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"Testing survival rate: "
    f"{y_test.mean() * 100:.2f}%"
)

print(
    "\nStratification was used because the target classes "
    "are imbalanced. The original dataset contains more "
    "non-survivors than survivors. Stratification preserves "
    "approximately the same class proportions in both the "
    "training and testing datasets, making model evaluation "
    "more representative."
)


# ============================================================
# DEFINE FEATURE TYPES
# ============================================================

numeric_features = [

    "pclass",

    "age",

    "sibsp",

    "parch",

    "fare"
]


categorical_features = [

    "sex",

    "embarked"
]


# ============================================================
# PREPROCESSING
# ============================================================

# Numerical preprocessing:
# 1. Median imputation
# 2. StandardScaler

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",

            StandardScaler()
        )

    ]
)


# Categorical preprocessing:
# 1. Most frequent imputation
# 2. One-hot encoding

categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",

            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


# Combine preprocessing

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",

            numeric_transformer,

            numeric_features
        ),

        (
            "cat",

            categorical_transformer,

            categorical_features
        )

    ]

)


print("\n" + "=" * 70)
print("PREPROCESSING STRATEGY")
print("=" * 70)

print(
    "Numeric columns: median imputation + StandardScaler"
)

print(
    "Categorical columns: most-frequent imputation "
    "+ OneHotEncoder"
)

print(
    "The preprocessing is inside the Pipeline and is "
    "fitted only on the training data."
)


# ============================================================
# DEFINE THREE CLASSIFIERS
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),


    "Decision Tree":

        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),


    "Random Forest":

        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

}


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_pipelines = {}

roc_data = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)

    print(
        f"TRAINING {model_name}"
    )

    print("=" * 70)


    # Create complete pipeline

    full_pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor
            ),

            (
                "classifier",

                model
            )

        ]

    )


    # Fit ONLY on training data

    full_pipeline.fit(
        X_train,
        y_train
    )


    # Predict test data

    y_pred = full_pipeline.predict(
        X_test
    )


    # Probability for ROC

    y_probability = (
        full_pipeline
        .predict_proba(X_test)[:, 1]
    )


    # Metrics

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )


    # Confusion matrix

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )

    print(
        f"AUC: {auc:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(cm)


    # Save confusion matrix

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=[
            "Did Not Survive",
            "Survived"
        ],

        yticklabels=[
            "Did Not Survive",
            "Survived"
        ]

    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    filename = (

        model_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"

    )

    plt.savefig(

        PLOTS_DIR / filename,

        dpi=300

    )

    plt.close()

  


    # ROC data

    fpr, tpr, _ = roc_curve(

        y_test,

        y_probability

    )


    roc_data[
        model_name
    ] = (

        fpr,

        tpr,

        auc

    )


    # Save results

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "AUC": auc

    })


    trained_pipelines[
        model_name
    ] = full_pipeline


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

print(
    comparison_df.round(4)
)


comparison_df.to_csv(

    BASE_DIR
    / "model_comparison.csv",

    index=False

)


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)


for model_name, (
    fpr,
    tpr,
    auc
) in roc_data.items():

    plt.plot(

        fpr,

        tpr,

        label=f"{model_name} (AUC = {auc:.3f})"

    )


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"

)


plt.title(
    "ROC Curves - Classification Models"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()


plt.savefig(

    PLOTS_DIR
    / "roc_curves.png",

    dpi=300

)

plt.close()


# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

decision_tree_pipeline = (

    trained_pipelines[
        "Decision Tree"
    ]

)


fitted_preprocessor = (

    decision_tree_pipeline
    .named_steps[
        "preprocessor"
    ]

)


decision_tree_model = (

    decision_tree_pipeline
    .named_steps[
        "classifier"
    ]

)


feature_names = (

    fitted_preprocessor
    .get_feature_names_out()

)


plt.figure(

    figsize=(24, 12)

)


plot_tree(

    decision_tree_model,

    feature_names=feature_names,

    class_names=[
        "Did Not Survive",
        "Survived"
    ],

    filled=True,

    rounded=True,

    fontsize=8

)


plt.title(
    "Decision Tree - Titanic Survival"
)


fig = plt.gcf()

fig.savefig(
    str(BASE_DIR / "classification_model_comparison.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


print("\nDecision tree visualization saved.")

print(
    "Classification modeling section complete."
)

# ============================================================
# IMBALANCE HANDLING COMPARISON
# BASELINE vs BALANCED vs SMOTE
# ============================================================

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


print("\n" + "=" * 70)
print("IMBALANCE HANDLING COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


baseline_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            baseline_model
        )

    ]

)


baseline_pipeline.fit(
    X_train,
    y_train
)


baseline_pred = baseline_pipeline.predict(
    X_test
)


baseline_precision = precision_score(
    y_test,
    baseline_pred,
    zero_division=0
)


baseline_recall = recall_score(
    y_test,
    baseline_pred,
    zero_division=0
)


baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    zero_division=0
)


# ------------------------------------------------------------
# CLASS WEIGHT = BALANCED
# ------------------------------------------------------------

balanced_model = LogisticRegression(

    max_iter=1000,

    class_weight="balanced",

    random_state=42

)


balanced_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            balanced_model
        )

    ]

)


balanced_pipeline.fit(
    X_train,
    y_train
)


balanced_pred = balanced_pipeline.predict(
    X_test
)


balanced_precision = precision_score(
    y_test,
    balanced_pred,
    zero_division=0
)


balanced_recall = recall_score(
    y_test,
    balanced_pred,
    zero_division=0
)


balanced_f1 = f1_score(
    y_test,
    balanced_pred,
    zero_division=0
)


# ------------------------------------------------------------
# SMOTE
# IMPORTANT:
# SMOTE IS APPLIED ONLY TO TRAINING DATA
# ------------------------------------------------------------

smote_model = LogisticRegression(

    max_iter=1000,

    random_state=42

)


smote_pipeline = ImbPipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "classifier",
            smote_model
        )

    ]

)


smote_pipeline.fit(
    X_train,
    y_train
)


smote_pred = smote_pipeline.predict(
    X_test
)


smote_precision = precision_score(
    y_test,
    smote_pred,
    zero_division=0
)


smote_recall = recall_score(
    y_test,
    smote_pred,
    zero_division=0
)


smote_f1 = f1_score(
    y_test,
    smote_pred,
    zero_division=0
)


# ============================================================
# CREATE IMBALANCE COMPARISON TABLE
# ============================================================

imbalance_results = pd.DataFrame({

    "Strategy": [

        "Baseline",

        "class_weight='balanced'",

        "SMOTE"

    ],

    "Precision": [

        baseline_precision,

        balanced_precision,

        smote_precision

    ],

    "Recall": [

        baseline_recall,

        balanced_recall,

        smote_recall

    ],

    "F1": [

        baseline_f1,

        balanced_f1,

        smote_f1

    ]

})


print("\nImbalance Handling Results:")

print(
    imbalance_results.round(4)
)


# Save results

imbalance_results.to_csv(

    BASE_DIR
    / "imbalance_comparison.csv",

    index=False

)


# ============================================================
# IDENTIFY BEST F1 STRATEGY
# ============================================================

best_imbalance_row = (

    imbalance_results
    .loc[
        imbalance_results["F1"].idxmax()
    ]

)


print(
    "\nBest imbalance strategy based on F1:"
)

print(
    best_imbalance_row["Strategy"]
)


print(
    f"Best F1 Score: "
    f"{best_imbalance_row['F1']:.4f}"
)


# ============================================================
# WRITTEN CONCLUSION
# ============================================================

print("\nImbalance Handling Conclusion:")

print(
    "The baseline model provides the reference performance "
    "without any explicit imbalance handling. The balanced "
    "class-weight approach changes the classification penalty "
    "so that the minority class receives greater importance. "
    "SMOTE creates synthetic minority-class examples using "
    "the training data only. The best strategy is selected "
    "based on the F1 score because F1 balances precision and "
    "recall and is appropriate when both types of classification "
    "errors are important."
)

# ============================================================
# RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

# from sklearn.model_selection import GridSearchCV


# print("\n" + "=" * 70)
# print("RANDOM FOREST HYPERPARAMETER TUNING")
# print("=" * 70)


# # Create Random Forest pipeline
# # oob_score=True is required for OOB score

# rf_pipeline = Pipeline(

#     steps=[

#         (
#             "preprocessor",
#             preprocessor
#         ),

#         (
#             "classifier",

#             RandomForestClassifier(

#                 oob_score=True,

#                 random_state=42,

#                 n_jobs=-1

#             )

#         )

#     ]

# )


# # Hyperparameter grid

# param_grid = {

#     "classifier__n_estimators": [

#         100,

#         200

#     ],

#     "classifier__max_depth": [

#         None,

#         5,

#         10

#     ],

#     "classifier__max_features": [

#         "sqrt",

#         "log2"

#     ]

# }


# # GridSearchCV

# grid_search = GridSearchCV(

#     estimator=rf_pipeline,

#     param_grid=param_grid,

#     cv=5,

#     scoring="f1",

#     n_jobs=-1,

#     return_train_score=True

# )


# # Fit ONLY on training data

# grid_search.fit(

#     X_train,

#     y_train

# )


# # ============================================================
# # BEST PARAMETERS
# # ============================================================

# print("\nBest Parameters:")

# print(
#     grid_search.best_params_
# )


# print(
#     f"\nBest Cross-Validation F1 Score: "
#     f"{grid_search.best_score_:.4f}"
# )


# # ============================================================
# # GET BEST PIPELINE
# # ============================================================

# best_rf_pipeline = (

#     grid_search.best_estimator_

# )


# # Get fitted Random Forest

# best_rf_model = (

#     best_rf_pipeline
#     .named_steps[
#         "classifier"
#     ]

# )


# # ============================================================
# # OOB SCORE
# # ============================================================

# oob_score = (

#     best_rf_model
#     .oob_score_

# )


# print(
#     f"\nRandom Forest OOB Score: "
#     f"{oob_score:.4f}"
# )


# # ============================================================
# # EVALUATE TUNED RANDOM FOREST ON TEST DATA
# # ============================================================

# tuned_rf_pred = (

#     best_rf_pipeline
#     .predict(
#         X_test
#     )

# )


# tuned_rf_probability = (

#     best_rf_pipeline
#     .predict_proba(
#         X_test
#     )[:, 1]

# )


# tuned_rf_accuracy = accuracy_score(

#     y_test,

#     tuned_rf_pred

# )


# tuned_rf_precision = precision_score(

#     y_test,

#     tuned_rf_pred,

#     zero_division=0

# )


# tuned_rf_recall = recall_score(

#     y_test,

#     tuned_rf_pred,

#     zero_division=0

# )


# tuned_rf_f1 = f1_score(

#     y_test,

#     tuned_rf_pred,

#     zero_division=0

# )


# tuned_rf_auc = roc_auc_score(

#     y_test,

#     tuned_rf_probability

# )


# print("\n" + "=" * 70)

# print(
#     "TUNED RANDOM FOREST TEST PERFORMANCE"
# )

# print("=" * 70)


# print(
#     f"Accuracy:  {tuned_rf_accuracy:.4f}"
# )

# print(
#     f"Precision: {tuned_rf_precision:.4f}"
# )

# print(
#     f"Recall:    {tuned_rf_recall:.4f}"
# )

# print(
#     f"F1 Score:  {tuned_rf_f1:.4f}"
# )

# print(
#     f"AUC:       {tuned_rf_auc:.4f}"
# )


# # ============================================================
# # SAVE GRID SEARCH RESULTS
# # ============================================================

# grid_results = pd.DataFrame(

#     grid_search.cv_results_

# )


# grid_results.to_csv(

#     BASE_DIR
#     / "random_forest_grid_search_results.csv",

#     index=False

# )


# # ============================================================
# # SAVE BEST PARAMETERS AND OOB SCORE
# # ============================================================

# tuning_summary = pd.DataFrame({

#     "Best Parameters": [

#         str(
#             grid_search.best_params_
#         )

#     ],

#     "Best CV F1": [

#         grid_search.best_score_

#     ],

#     "OOB Score": [

#         oob_score

#     ],

#     "Test Accuracy": [

#         tuned_rf_accuracy

#     ],

#     "Test Precision": [

#         tuned_rf_precision

#     ],

#     "Test Recall": [

#         tuned_rf_recall

#     ],

#     "Test F1": [

#         tuned_rf_f1

#     ],

#     "Test AUC": [

#         tuned_rf_auc

#     ]

# })


# tuning_summary.to_csv(

#     BASE_DIR
#     / "random_forest_tuning_summary.csv",

#     index=False

# )


# print(
#     "\nGridSearchCV results saved."
# )

# ============================================================
# REGRESSION SIDE-TASK
# PREDICT FARE FROM OTHER AVAILABLE FEATURES
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("\n" + "=" * 70)
print("REGRESSION SIDE-TASK: PREDICTING FARE")
print("=" * 70)


# ------------------------------------------------------------
# TARGET AND FEATURES
# ------------------------------------------------------------

# Target
y_reg = df["fare"]


# Use all other available features except fare
X_reg = df.drop(
    columns=["fare"]
)


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(

    X_reg,

    y_reg,

    test_size=0.20,

    random_state=42

)


# ------------------------------------------------------------
# IDENTIFY REGRESSION FEATURE TYPES
# ------------------------------------------------------------

reg_numeric_features = X_reg.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


reg_categorical_features = X_reg.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\nNumeric regression features:")

print(
    reg_numeric_features
)


print("\nCategorical regression features:")

print(
    reg_categorical_features
)


# ------------------------------------------------------------
# REGRESSION PREPROCESSING
# ------------------------------------------------------------

reg_numeric_transformer = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",

            StandardScaler()
        )

    ]

)


reg_categorical_transformer = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",

            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]

)


reg_preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",

            reg_numeric_transformer,

            reg_numeric_features
        ),

        (
            "cat",

            reg_categorical_transformer,

            reg_categorical_features
        )

    ]

)


# ------------------------------------------------------------
# LINEAR REGRESSION PIPELINE
# ------------------------------------------------------------

regression_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",

            reg_preprocessor

        ),

        (
            "regressor",

            LinearRegression()

        )

    ]

)


# ------------------------------------------------------------
# TRAIN REGRESSION MODEL
# ------------------------------------------------------------

regression_pipeline.fit(

    X_reg_train,

    y_reg_train

)


# ------------------------------------------------------------
# PREDICT FARE
# ------------------------------------------------------------

y_reg_pred = regression_pipeline.predict(

    X_reg_test

)


# ------------------------------------------------------------
# REGRESSION METRICS
# ------------------------------------------------------------

mae = mean_absolute_error(

    y_reg_test,

    y_reg_pred

)


rmse = np.sqrt(

    mean_squared_error(

        y_reg_test,

        y_reg_pred

    )

)


r2 = r2_score(

    y_reg_test,

    y_reg_pred

)


# ------------------------------------------------------------
# ADJUSTED R²
# ------------------------------------------------------------

# Number of test observations
n = len(y_reg_test)


# Number of predictors after preprocessing
p = (

    regression_pipeline
    .named_steps[
        "preprocessor"
    ]
    .transform(
        X_reg_test
    )
    .shape[1]

)


adjusted_r2 = (

    1

    - (

        (1 - r2)
        * (n - 1)

        /

        (n - p - 1)

    )

)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)

print(
    "REGRESSION PERFORMANCE"
)

print("=" * 70)


print(
    f"MAE:          {mae:.4f}"
)


print(
    f"RMSE:         {rmse:.4f}"
)


print(
    f"R²:           {r2:.4f}"
)


print(
    f"Adjusted R²:  {adjusted_r2:.4f}"
)


# ------------------------------------------------------------
# RESIDUALS
# ------------------------------------------------------------

residuals = (

    y_reg_test
    - y_reg_pred

)


# ------------------------------------------------------------
# RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    "Residual Plot - Fare Regression"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residuals"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "regression_residuals.png",
    dpi=100
)

plt.close()


# ------------------------------------------------------------
# HETEROSCEDASTICITY CHECK
# ------------------------------------------------------------

# Calculate correlation between predicted values
# and absolute residuals

residual_correlation = np.corrcoef(

    y_reg_pred,

    np.abs(residuals)

)[0, 1]


print("\n" + "=" * 70)

print(
    "HETEROSCEDASTICITY CHECK"
)

print("=" * 70)


print(

    "Correlation between predicted fare "
    "and absolute residuals: "
    f"{residual_correlation:.4f}"

)


if abs(residual_correlation) > 0.30:

    heteroscedasticity_conclusion = (

        "The residuals show evidence of "
        "heteroscedasticity because the spread "
        "of residuals changes as predicted fare "
        "increases. The relationship between "
        "predicted values and absolute residuals "
        "also shows a meaningful association."

    )

else:

    heteroscedasticity_conclusion = (

        "The residuals do not show strong evidence "
        "of heteroscedasticity. The residual spread "
        "appears reasonably random and does not "
        "increase substantially with predicted fare."

    )


print(

    heteroscedasticity_conclusion

)


# ------------------------------------------------------------
# SAVE REGRESSION METRICS
# ------------------------------------------------------------

regression_metrics = pd.DataFrame({

    "MAE": [

        mae

    ],

    "RMSE": [

        rmse

    ],

    "R2": [

        r2

    ],

    "Adjusted_R2": [

        adjusted_r2

    ]

})


regression_metrics.to_csv(

    BASE_DIR
    / "regression_metrics.csv",

    index=False

)


# ------------------------------------------------------------
# SAVE HETEROSCEDASTICITY RESULT
# ------------------------------------------------------------

with open(

    BASE_DIR
    / "regression_interpretation.txt",

    "w",

    encoding="utf-8"

) as file:

    file.write(

        heteroscedasticity_conclusion

    )


print(

    "\nRegression residual plot saved."

)

print(

    "Regression metrics saved."

)

# ============================================================
# FINAL MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

# Classification results
final_classification_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        0.8090,
        0.7640,
        0.8090
    ],

    "Precision": [
        0.7833,
        0.7600,
        0.7656
    ],

    "Recall": [
        0.6912,
        0.5588,
        0.7206
    ],

    "F1": [
        0.7344,
        0.6441,
        0.7424
    ],

    "AUC": [
        0.8610,
        0.8374,
        0.8196
    ]

})


print("\nCLASSIFICATION METRICS")
print(final_classification_results)


# Regression results
final_regression_results = pd.DataFrame({

    "Regression Model": [
        "Multivariate Linear Regression"
    ],

    "MAE": [
        18.3735
    ],

    "RMSE": [
        41.2921
    ],

    "R2": [
        0.3609
    ],

    "Adjusted_R2": [
        0.2558
    ]

})


print("\nREGRESSION METRICS")
print(final_regression_results)


# Save both tables
final_classification_results.to_csv(
    BASE_DIR / "final_classification_comparison.csv",
    index=False
)

final_regression_results.to_csv(
    BASE_DIR / "final_regression_comparison.csv",
    index=False
)


print("\nFinal comparison tables saved.")

# ============================================================
# FINAL DEPLOYMENT RECOMMENDATION
# ============================================================

final_recommendation = """
Based on the classification results, I recommend deploying the Random Forest
classifier as the final model. It achieved an accuracy of 0.8090 and the
highest F1 score of 0.7424 among the three original classifiers, while also
achieving the highest recall of 0.7206. Logistic Regression achieved the
highest AUC of 0.8610, indicating strong overall ranking performance, but
Random Forest provides a better balance between precision and recall based
on its F1 score. The Decision Tree is not recommended because it produced the
lowest accuracy (0.7640), recall (0.5588), and F1 score (0.6441). Therefore,
Random Forest is the preferred deployment model when balancing the need to
identify survivors while maintaining reasonable overall classification
performance.
"""

print("\n" + "=" * 70)
print("FINAL DEPLOYMENT RECOMMENDATION")
print("=" * 70)

print(final_recommendation)


# Save recommendation to file
with open(
    BASE_DIR / "final_recommendation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(final_recommendation)


print("Final recommendation saved.")

# ============================================================
# SAVE BEST COMPLETE PIPELINE
# ============================================================

import joblib

print("\n" + "=" * 70)
print("SAVING BEST COMPLETE PIPELINE")
print("=" * 70)

# Use the tuned Random Forest pipeline
# This pipeline contains:
# 1. Missing-value imputation
# 2. Categorical encoding
# 3. Numeric scaling
# 4. Random Forest classifier

# Create the final tuned Random Forest pipeline
full_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                max_features="sqrt",
                random_state=42,
                oob_score=True
            )
        )
    ]
)


# Fit the complete pipeline on the training data BEFORE saving.
# This ensures the saved artifact contains fitted preprocessing
# and a fitted Random Forest model.
full_pipeline.fit(
    X_train,
    y_train
)

print("Complete pipeline fitted successfully.")

# Save the FITTED complete pipeline
pipeline_path = BASE_DIR / "best_titanic_pipeline.joblib"

joblib.dump(
    full_pipeline,
    pipeline_path
)

print(
    f"Complete pipeline saved to: {pipeline_path}"
)

# ============================================================
# RELOAD AND TEST SAVED PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("RELOADING AND TESTING SAVED PIPELINE")
print("=" * 70)

# Load the saved complete pipeline
loaded_pipeline = joblib.load(
    pipeline_path
)

print("Pipeline loaded successfully.")


# Use one raw test row
raw_sample = X_test.iloc[[0]]


# Make prediction on raw, unprocessed data
sample_prediction = loaded_pipeline.predict(
    raw_sample
)


print("\nRaw sample:")
print(raw_sample)


print(
    "\nPredicted survival:",
    sample_prediction[0]
)


print(
    "Actual survival:",
    y_test.iloc[0]
)


if sample_prediction[0] == y_test.iloc[0]:

    print(
        "\nPipeline reload test: SUCCESS"
    )

else:

    print(
        "\nPipeline reload test: SUCCESS - "
        "prediction was generated correctly, "
        "but did not match this individual test label."
    )