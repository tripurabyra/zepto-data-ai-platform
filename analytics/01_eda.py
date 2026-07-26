import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# TASK 1: LOAD TITANIC DATASET
# IMPORTANT: THIS IS THE ONLY sns.load_dataset() CALL
# ============================================================

print("=" * 60)
print("LOADING TITANIC DATASET")
print("=" * 60)

df = sns.load_dataset("titanic")


# ============================================================
# SAVE OFFLINE FALLBACK
# ============================================================

csv_path = BASE_DIR / "titanic.csv"

df.to_csv(csv_path, index=False)

print(f"\nDataset saved to: {csv_path}")


# ============================================================
# PROFILE DATASET
# ============================================================

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)


print("\n" + "=" * 60)
print("DATASET INFO")
print("=" * 60)

df.info()


print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe(include="all"))


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_count = df.isnull().sum()
missing_percentage = (df.isnull().mean() * 100)

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

missing_report = missing_report[
    missing_report["Missing Count"] > 0
]

print(missing_report)

# ============================================================
# TASK 2: MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE HANDLING")
print("=" * 60)

# Make a copy so the original loaded dataset remains unchanged
cleaned_df = df.copy()


# ------------------------------------------------------------
# AGE: 19.87% MISSING
# 5%–30% MISSING -> IMPUTE
# ------------------------------------------------------------

age_missing_pct = df["age"].isnull().mean() * 100

print(f"\nAge missing percentage: {age_missing_pct:.4f}%")
print("Strategy: Median imputation.")
print("Reason: Missing percentage is between 5% and 30%.")


# Median imputation
cleaned_df["age"] = cleaned_df["age"].fillna(
    cleaned_df["age"].median()
)


# ------------------------------------------------------------
# EMBARKED: 0.22% MISSING
# UNDER 5% -> DROP ROWS
# ------------------------------------------------------------

embarked_missing_pct = df["embarked"].isnull().mean() * 100

print(f"\nEmbarked missing percentage: {embarked_missing_pct:.4f}%")
print("Strategy: Drop rows with missing embarked values.")
print("Reason: Missing percentage is below 5%.")


cleaned_df = cleaned_df.dropna(
    subset=["embarked"]
)


# ------------------------------------------------------------
# EMBARK_TOWN: 0.22% MISSING
# UNDER 5% -> DROP ROWS
# ------------------------------------------------------------

embark_town_missing_pct = df["embark_town"].isnull().mean() * 100

print(
    f"\nEmbark_town missing percentage: "
    f"{embark_town_missing_pct:.4f}%"
)

print("Strategy: Drop rows with missing embark_town values.")
print("Reason: Missing percentage is below 5%.")


cleaned_df = cleaned_df.dropna(
    subset=["embark_town"]
)


# ------------------------------------------------------------
# DECK: 77.22% MISSING
# HIGH MISSINGNESS -> DROP COLUMN
# ------------------------------------------------------------

deck_missing_pct = df["deck"].isnull().mean() * 100

print(f"\nDeck missing percentage: {deck_missing_pct:.4f}%")
print("Strategy: Drop the deck column.")
print(
    "Reason: More than 30% of values are missing, "
    "so imputation would be unreliable."
)


cleaned_df = cleaned_df.drop(
    columns=["deck"]
)


# ------------------------------------------------------------
# VERIFY CLEANED DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLEANED DATASET INFORMATION")
print("=" * 60)

print("Original shape:", df.shape)
print("Cleaned shape:", cleaned_df.shape)

print("\nRemaining missing values:")

remaining_missing = cleaned_df.isnull().sum()

print(
    remaining_missing[
        remaining_missing > 0
    ]
)


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

cleaned_csv_path = BASE_DIR / "titanic.csv"

cleaned_df.to_csv(
    cleaned_csv_path,
    index=False
)

print(
    f"\nCleaned dataset saved to: "
    f"{cleaned_csv_path}"
)
# ============================================================
# SAVE MISSING VALUE REPORT
# ============================================================

missing_report.to_csv(
    BASE_DIR / "missing_value_report.csv"
)

print("\nMissing value report saved.")


print("\n" + "=" * 60)
print("EDA STEP 1 COMPLETE")
print("=" * 60)


# ============================================================
# TASK 3: UNIVARIATE ANALYSIS
# AGE AND FARE
# ============================================================

print("\n" + "=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)


# ============================================================
# AGE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    cleaned_df["age"],
    bins=30,
    kde=True
)

plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "age_histogram.png",
    dpi=300
)

plt.show()


# ============================================================
# AGE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=cleaned_df["age"]
)

plt.title("Box Plot of Passenger Age")
plt.xlabel("Age")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "age_boxplot.png",
    dpi=300
)

plt.show()


# ============================================================
# FARE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    cleaned_df["fare"],
    bins=30,
    kde=True
)

plt.title("Distribution of Passenger Fare")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "fare_histogram.png",
    dpi=300
)

plt.show()


# ============================================================
# FARE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=cleaned_df["fare"]
)

plt.title("Box Plot of Passenger Fare")
plt.xlabel("Fare")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "fare_boxplot.png",
    dpi=300
)

plt.show()


# ============================================================
# IQR OUTLIER FUNCTION
# ============================================================

def calculate_iqr_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": len(outliers)
    }


# ============================================================
# AGE IQR OUTLIERS
# ============================================================

age_outlier_results = calculate_iqr_outliers(
    cleaned_df["age"]
)

print("\nAGE IQR OUTLIER ANALYSIS")

for key, value in age_outlier_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# FARE IQR OUTLIERS
# ============================================================

fare_outlier_results = calculate_iqr_outliers(
    cleaned_df["fare"]
)

print("\nFARE IQR OUTLIER ANALYSIS")

for key, value in fare_outlier_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# FARE STATISTICS
# ============================================================

fare_mean = cleaned_df["fare"].mean()

fare_median = cleaned_df["fare"].median()

fare_mode = cleaned_df["fare"].mode()


print("\n" + "=" * 60)
print("FARE STATISTICS")
print("=" * 60)

print(
    f"Mean: {fare_mean:.4f}"
)

print(
    f"Median: {fare_median:.4f}"
)

print(
    f"Mode: {fare_mode.tolist()}"
)


# ============================================================
# FARE SKEWNESS INTERPRETATION
# ============================================================

print("\nFARE DISTRIBUTION INTERPRETATION")

if fare_mean > fare_median > fare_mode.iloc[0]:

    print(
        "Fare is right-skewed."
    )

    print(
        "The mean is greater than the median, "
        "and the median is greater than the mode. "
        "This ordering indicates a positive/right-skewed distribution."
    )

elif fare_mean < fare_median < fare_mode.iloc[0]:

    print(
        "Fare is left-skewed."
    )

    print(
        "The mean is less than the median, "
        "and the median is less than the mode. "
        "This ordering indicates a negative/left-skewed distribution."
    )

else:

    print(
        "Fare does not follow a perfectly symmetric "
        "mean-median-mode ordering."
    )

    print(
        "The histogram and box plot should be considered "
        "along with the mean, median, and mode."
    )

    # ============================================================
# TASK 4: BIVARIATE ANALYSIS
# SURVIVAL RATE BY SEX, PCLASS, AND SEX + PCLASS
# ============================================================

print("\n" + "=" * 60)
print("BIVARIATE SURVIVAL ANALYSIS")
print("=" * 60)


# ============================================================
# SURVIVAL RATE BY SEX
# ============================================================

print("\nSURVIVAL RATE BY SEX")

survival_by_sex = (
    cleaned_df
    .groupby("sex")["survived"]
    .mean()
    .sort_values(ascending=False)
)

print(
    (survival_by_sex * 100).round(2)
)


# Boolean masking examples
female_data = cleaned_df[
    cleaned_df["sex"] == "female"
]

male_data = cleaned_df[
    cleaned_df["sex"] == "male"
]

female_survival_rate = (
    female_data["survived"].mean() * 100
)

male_survival_rate = (
    male_data["survived"].mean() * 100
)

print(
    f"\nFemale survival rate: "
    f"{female_survival_rate:.2f}%"
)

print(
    f"Male survival rate: "
    f"{male_survival_rate:.2f}%"
)


# ============================================================
# SURVIVAL RATE BY PCLASS
# ============================================================

print("\nSURVIVAL RATE BY PCLASS")

survival_by_pclass = (
    cleaned_df
    .groupby("pclass")["survived"]
    .mean()
    .sort_index()
)

print(
    (survival_by_pclass * 100).round(2)
)


# Boolean masking for passenger classes

first_class = cleaned_df[
    cleaned_df["pclass"] == 1
]

second_class = cleaned_df[
    cleaned_df["pclass"] == 2
]

third_class = cleaned_df[
    cleaned_df["pclass"] == 3
]

print(
    f"\n1st class survival rate: "
    f"{first_class['survived'].mean() * 100:.2f}%"
)

print(
    f"2nd class survival rate: "
    f"{second_class['survived'].mean() * 100:.2f}%"
)

print(
    f"3rd class survival rate: "
    f"{third_class['survived'].mean() * 100:.2f}%"
)


# ============================================================
# SURVIVAL RATE BY SEX + PCLASS
# ============================================================

print("\nSURVIVAL RATE BY SEX AND PCLASS")

survival_by_sex_pclass = (
    cleaned_df
    .groupby(["sex", "pclass"])["survived"]
    .mean()
)

print(
    (survival_by_sex_pclass * 100).round(2)
)


# ============================================================
# BOOLEAN MASKING WITH & COMBINATION
# ============================================================

print("\nBOOLEAN MASKING EXAMPLES")


# Female AND first class
female_first = cleaned_df[
    (cleaned_df["sex"] == "female") &
    (cleaned_df["pclass"] == 1)
]

# Female AND third class
female_third = cleaned_df[
    (cleaned_df["sex"] == "female") &
    (cleaned_df["pclass"] == 3)
]

# Male AND first class
male_first = cleaned_df[
    (cleaned_df["sex"] == "male") &
    (cleaned_df["pclass"] == 1)
]

# Male AND third class
male_third = cleaned_df[
    (cleaned_df["sex"] == "male") &
    (cleaned_df["pclass"] == 3)
]


print(
    f"Female + 1st Class: "
    f"{female_first['survived'].mean() * 100:.2f}%"
)

print(
    f"Female + 3rd Class: "
    f"{female_third['survived'].mean() * 100:.2f}%"
)

print(
    f"Male + 1st Class: "
    f"{male_first['survived'].mean() * 100:.2f}%"
)

print(
    f"Male + 3rd Class: "
    f"{male_third['survived'].mean() * 100:.2f}%"
)


# ============================================================
# BOOLEAN MASKING WITH OR |
# ============================================================

# Example: passengers who were either female OR first class

female_or_first = cleaned_df[
    (cleaned_df["sex"] == "female") |
    (cleaned_df["pclass"] == 1)
]

print(
    "\nPassengers who were female OR first class: "
    f"{len(female_or_first)}"
)


# ============================================================
# PLOT 1: SURVIVAL BY SEX
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=cleaned_df,
    x="sex",
    y="survived"
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "survival_by_sex.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 2: SURVIVAL BY PCLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "survival_by_pclass.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 3: SURVIVAL BY SEX AND PCLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Passenger Class and Sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "survival_sex_pclass.png",
    dpi=300
)

plt.show()
# ============================================================
# TASK 5: CORRELATION MATRIX
# EXACTLY SIX REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)


# Required columns ONLY
correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]


# Create correlation dataframe
correlation_df = cleaned_df[
    correlation_columns
]


# Calculate correlation matrix
correlation_matrix = correlation_df.corr()


print("\n6 x 6 CORRELATION MATRIX")
print(correlation_matrix.round(4))


# ============================================================
# FIND TWO STRONGEST CORRELATIONS
# ============================================================
# ============================================================
# FIND TWO STRONGEST CORRELATIONS
# ============================================================

correlation_pairs = []

features = correlation_matrix.columns.tolist()

# Only examine the upper triangle.
# This automatically excludes:
# - self-correlations
# - duplicate pairs

for i in range(len(features)):

    for j in range(i + 1, len(features)):

        feature1 = features[i]
        feature2 = features[j]

        correlation_value = correlation_matrix.loc[
            feature1,
            feature2
        ]

        absolute_value = abs(
            correlation_value
        )

        correlation_pairs.append(
            (
                feature1,
                feature2,
                correlation_value,
                absolute_value
            )
        )


# Sort by absolute correlation
correlation_pairs = sorted(
    correlation_pairs,
    key=lambda x: x[3],
    reverse=True
)


# Get top two
top_two_pairs = correlation_pairs[:2]


# ============================================================
# PRINT TWO STRONGEST CORRELATIONS
# ============================================================

print("\n" + "=" * 60)
print("TWO STRONGEST CORRELATIONS")
print("=" * 60)

for rank, (
    feature1,
    feature2,
    correlation_value,
    absolute_value
) in enumerate(
    top_two_pairs,
    start=1
):

    print(
        f"{rank}. "
        f"{feature1} vs {feature2}: "
        f"{correlation_value:.4f} "
        f"(absolute correlation: "
        f"{absolute_value:.4f})"
    )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title(
    "Correlation Matrix of Titanic Numeric Features"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE CORRELATION MATRIX
# ============================================================

correlation_matrix.to_csv(
    BASE_DIR / "correlation_matrix.csv"
)

print(
    "\nCorrelation matrix saved to "
    "correlation_matrix.csv"
)

# ============================================================
# TASK 6: MULTIVARIATE DATA STORY
# ADDITIONAL VISUALIZATIONS
# ============================================================


# ============================================================
# CHART 4: SURVIVAL BY AGE
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=cleaned_df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "age_by_survival.png",
    dpi=300
)

plt.show()


# ============================================================
# CHART 5: SURVIVAL BY FARE
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=cleaned_df,
    x="survived",
    y="fare"
)

plt.title("Fare Distribution by Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "fare_by_survival.png",
    dpi=300
)

plt.show()


# ============================================================
# CHART 6: SURVIVAL BY SEX AND CLASS
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Passenger Class and Sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "multivariate_survival_story.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 60)
print("MULTIVARIATE DATA STORY CHARTS CREATED")
print("=" * 60)

print("1. Survival by Sex")
print("2. Survival by Passenger Class")
print("3. Survival by Sex and Passenger Class")
print("4. Correlation Heatmap")
print("5. Age Distribution by Survival")
print("6. Fare Distribution by Survival")

# ============================================================
# TASK 7: EXPLORATORY STANDARDIZATION CHECK
# AGE AND FARE
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZATION CHECK")
print("=" * 60)


# ------------------------------------------------------------
# BEFORE STANDARDIZATION
# ------------------------------------------------------------

print("\nBEFORE STANDARDIZATION")

print(
    "\nAge:"
)

print(
    f"Mean: {cleaned_df['age'].mean():.6f}"
)

print(
    f"Standard Deviation: "
    f"{cleaned_df['age'].std():.6f}"
)


print(
    "\nFare:"
)

print(
    f"Mean: {cleaned_df['fare'].mean():.6f}"
)

print(
    f"Standard Deviation: "
    f"{cleaned_df['fare'].std():.6f}"
)


# ------------------------------------------------------------
# STANDARDIZE USING Z-SCORE
# z = (x - mean) / std
# ------------------------------------------------------------

standardized_df = cleaned_df.copy()


standardized_df["age_z"] = (
    standardized_df["age"]
    - standardized_df["age"].mean()
) / standardized_df["age"].std()


standardized_df["fare_z"] = (
    standardized_df["fare"]
    - standardized_df["fare"].mean()
) / standardized_df["fare"].std()


# ------------------------------------------------------------
# AFTER STANDARDIZATION
# ------------------------------------------------------------

print("\nAFTER STANDARDIZATION")

print(
    "\nStandardized Age:"
)

print(
    f"Mean: "
    f"{standardized_df['age_z'].mean():.6f}"
)

print(
    f"Standard Deviation: "
    f"{standardized_df['age_z'].std():.6f}"
)


print(
    "\nStandardized Fare:"
)

print(
    f"Mean: "
    f"{standardized_df['fare_z'].mean():.6f}"
)

print(
    f"Standard Deviation: "
    f"{standardized_df['fare_z'].std():.6f}"
)


# ============================================================
# BEFORE / AFTER SUMMARY TABLE
# ============================================================

standardization_summary = pd.DataFrame({

    "Variable": [
        "Age",
        "Fare"
    ],

    "Before Mean": [
        cleaned_df["age"].mean(),
        cleaned_df["fare"].mean()
    ],

    "Before Std": [
        cleaned_df["age"].std(),
        cleaned_df["fare"].std()
    ],

    "After Mean": [
        standardized_df["age_z"].mean(),
        standardized_df["fare_z"].mean()
    ],

    "After Std": [
        standardized_df["age_z"].std(),
        standardized_df["fare_z"].std()
    ]

})


print(
    "\nSTANDARDIZATION SUMMARY"
)

print(
    standardization_summary.round(6)
)


# Save summary
standardization_summary.to_csv(
    BASE_DIR / "standardization_summary.csv",
    index=False
)


# ============================================================
# STANDARDIZATION COMPARISON PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# Age
axes[0].hist(
    cleaned_df["age"],
    bins=30,
    alpha=0.7,
    label="Original Age"
)

axes[0].hist(
    standardized_df["age_z"],
    bins=30,
    alpha=0.7,
    label="Standardized Age"
)

axes[0].set_title(
    "Age Before and After Standardization"
)

axes[0].set_xlabel(
    "Value"
)

axes[0].set_ylabel(
    "Frequency"
)

axes[0].legend()


# Fare
axes[1].hist(
    cleaned_df["fare"],
    bins=30,
    alpha=0.7,
    label="Original Fare"
)

axes[1].hist(
    standardized_df["fare_z"],
    bins=30,
    alpha=0.7,
    label="Standardized Fare"
)

axes[1].set_title(
    "Fare Before and After Standardization"
)

axes[1].set_xlabel(
    "Value"
)

axes[1].set_ylabel(
    "Frequency"
)

axes[1].legend()


plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "standardization_comparison.png",
    dpi=300
)

plt.show()