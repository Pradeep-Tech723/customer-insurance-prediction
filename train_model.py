import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs("model", exist_ok=True)
os.makedirs("graphs", exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/Social_Network_Ads.csv")

print("\n==============================================")
print("DATASET INFORMATION")
print("==============================================")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# DATA PREPARATION
# ============================================================

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# CREATE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(random_state=42),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
}


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

results = []

confusion_matrices = {}

trained_models = {}


print("\n==============================================")
print("MODEL COMPARISON")
print("==============================================")


for name, model in models.items():

    # Models based on distance/margin benefit from scaling.
    # Tree-based models do not require scaling, but using
    # the same prepared data keeps the comparison consistent.

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    confusion_matrices[name] = cm

    trained_models[name] = model

    results.append({

        "Model": name,

        "Accuracy": accuracy * 100,

        "Precision": precision * 100,

        "Recall": recall * 100,

        "F1 Score": f1 * 100

    })

    print("\n", name)

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n==============================================")
print("FINAL COMPARISON TABLE")
print("==============================================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_accuracy = results_df.iloc[0]["Accuracy"]


print("\n==============================================")
print("BEST MODEL")
print("==============================================")

print(
    f"Best Model: {best_model_name}"
)

print(
    f"Accuracy: {best_accuracy:.2f}%"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

with open(
    "model/best_model.pkl",
    "wb"
) as file:

    pickle.dump(
        best_model,
        file
    )


# ============================================================
# SAVE SCALER
# ============================================================

with open(
    "model/scaler.pkl",
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "model/model_results.pkl",
    "wb"
) as file:

    pickle.dump(
        results_df,
        file
    )


# ============================================================
# SAVE ACCURACY
# ============================================================

with open(
    "model/accuracy.pkl",
    "wb"
) as file:

    pickle.dump(
        best_accuracy,
        file
    )


# ============================================================
# GRAPH 1 - AGE VS PURCHASE - IMPROVED
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Age",
    hue="Purchased",
    bins=15,
    kde=True,
    element="step",
    stat="count",
    common_norm=False
)

plt.title(
    "Age Distribution by Purchase Decision",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Customer Age",
    fontsize=12
)

plt.ylabel(
    "Number of Customers",
    fontsize=12
)

plt.legend(
    title="Purchased",
    labels=["Yes", "No"]
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "graphs/age_purchase.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# GRAPH 2 - SALARY VS PURCHASE - IMPROVED
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="EstimatedSalary",
    hue="Purchased",
    bins=15,
    kde=True,
    element="step",
    stat="count",
    common_norm=False
)

plt.title(
    "Estimated Salary Distribution by Purchase Decision",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Estimated Salary",
    fontsize=12
)

plt.ylabel(
    "Number of Customers",
    fontsize=12
)

plt.legend(
    title="Purchased",
    labels=["Yes", "No"]
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "graphs/salary_purchase.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# GRAPH 3
# MODEL ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title(
    "Classification Algorithm Accuracy Comparison"
)

plt.xlabel("Algorithm")

plt.ylabel("Accuracy (%)")

plt.xticks(
    rotation=25
)

plt.ylim(
    0,
    100
)

plt.tight_layout()

plt.savefig(
    "graphs/model_comparison.png"
)

plt.close()


# ============================================================
# GRAPH 4
# ALL METRICS COMPARISON
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

x = np.arange(
    len(results_df["Model"])
)

width = 0.2

plt.figure(figsize=(12, 7))

for i, metric in enumerate(metrics):

    plt.bar(
        x + (i - 1.5) * width,
        results_df[metric],
        width,
        label=metric
    )

plt.xticks(
    x,
    results_df["Model"],
    rotation=25
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Accuracy, Precision, Recall and F1 Score"
)

plt.legend()

plt.ylim(
    0,
    100
)

plt.tight_layout()

plt.savefig(
    "graphs/metrics_comparison.png"
)

plt.close()


# ============================================================
# GRAPH 5
# CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 9)
)

axes = axes.flatten()

for i, name in enumerate(
    models.keys()
):

    sns.heatmap(
        confusion_matrices[name],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[i]
    )

    axes[i].set_title(
        name
    )

    axes[i].set_xlabel(
        "Predicted"
    )

    axes[i].set_ylabel(
        "Actual"
    )


# Remove empty sixth plot

axes[5].axis("off")

plt.tight_layout()

plt.savefig(
    "graphs/confusion_matrices.png"
)

plt.close()


# ============================================================
# QUESTION 1 AND QUESTION 2 SCENARIOS
# ============================================================

# For "No Salary", a prediction cannot technically be made
# without a numeric salary.
#
# To satisfy the assignment while remaining transparent,
# we use the dataset median salary as an explicit assumption.

median_salary = df["EstimatedSalary"].median()

print("\n==============================================")
print("NO-SALARY ASSUMPTION")
print("==============================================")

print(
    f"Median salary used when salary is not provided: "
    f"{median_salary:.0f}"
)


scenarios = [

    {
        "Question": "Q1",
        "Age": 30,
        "Salary": 87000,
        "Description": "Age 30, Salary 87,000"
    },

    {
        "Question": "Q1",
        "Age": 40,
        "Salary": median_salary,
        "Description": "Age 40, No Salary (median assumed)"
    },

    {
        "Question": "Q1",
        "Age": 40,
        "Salary": 100000,
        "Description": "Age 40, Salary 100,000"
    },

    {
        "Question": "Q1",
        "Age": 50,
        "Salary": median_salary,
        "Description": "Age 50, No Salary (median assumed)"
    },

    {
        "Question": "Q2",
        "Age": 18,
        "Salary": median_salary,
        "Description": "Age 18, No Salary (median assumed)"
    },

    {
        "Question": "Q2",
        "Age": 22,
        "Salary": 600000,
        "Description": "Age 22, Salary 600,000"
    },

    {
        "Question": "Q2",
        "Age": 35,
        "Salary": 2500000,
        "Description": "Age 35, Salary 2,500,000"
    },

    {
        "Question": "Q2",
        "Age": 60,
        "Salary": 100000000,
        "Description": "Age 60, Salary 100,000,000"
    }

]


print("\n==============================================")
print("SCENARIO PREDICTIONS")
print("==============================================")


scenario_results = []


for scenario in scenarios:

    data = np.array([
        [
            scenario["Age"],
            scenario["Salary"]
        ]
    ])

    data_scaled = scaler.transform(
        data
    )

    prediction = best_model.predict(
        data_scaled
    )[0]

    probability = best_model.predict_proba(
        data_scaled
    )[0][1] * 100

    result_text = (
        "Likely to purchase"
        if prediction == 1
        else
        "Unlikely to purchase"
    )

    print("\n" + scenario["Description"])

    print(
        f"Prediction: {result_text}"
    )

    print(
        f"Purchase probability: "
        f"{probability:.2f}%"
    )

    scenario_results.append({

        "Question":
            scenario["Question"],

        "Scenario":
            scenario["Description"],

        "Age":
            scenario["Age"],

        "Salary":
            scenario["Salary"],

        "Prediction":
            result_text,

        "Probability":
            probability

    })


scenario_df = pd.DataFrame(
    scenario_results
)


# ============================================================
# SAVE SCENARIO RESULTS
# ============================================================

with open(
    "model/scenario_results.pkl",
    "wb"
) as file:

    pickle.dump(
        scenario_df,
        file
    )


# ============================================================
# HYPOTHESIS TESTING
# ============================================================

print("\n==============================================")
print("HYPOTHESIS TESTING")
print("==============================================")


# Hypothesis 1:
# Higher salary should generally increase purchase probability.

test_ages = [25, 35, 45, 55]

salary_low = df["EstimatedSalary"].quantile(0.25)

salary_high = df["EstimatedSalary"].quantile(0.75)

hypothesis_results = []


for age in test_ages:

    low_data = np.array([
        [age, salary_low]
    ])

    high_data = np.array([
        [age, salary_high]
    ])

    low_scaled = scaler.transform(
        low_data
    )

    high_scaled = scaler.transform(
        high_data
    )

    low_probability = (
        best_model.predict_proba(
            low_scaled
        )[0][1] * 100
    )

    high_probability = (
        best_model.predict_proba(
            high_scaled
        )[0][1] * 100
    )

    hypothesis_results.append({

        "Age": age,

        "Low Salary Probability":
            low_probability,

        "High Salary Probability":
            high_probability

    })

    print(
        f"Age {age}: "
        f"Low salary = {low_probability:.2f}% | "
        f"High salary = {high_probability:.2f}%"
    )


hypothesis_df = pd.DataFrame(
    hypothesis_results
)


# ============================================================
# HYPOTHESIS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    hypothesis_df["Age"],
    hypothesis_df["Low Salary Probability"],
    marker="o",
    label="Lower Salary"
)

plt.plot(
    hypothesis_df["Age"],
    hypothesis_df["High Salary Probability"],
    marker="o",
    label="Higher Salary"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Purchase Probability (%)"
)

plt.title(
    "Hypothesis Test: Salary Effect on Purchase Probability"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "graphs/hypothesis_analysis.png"
)

plt.close()


# ============================================================
# SAVE HYPOTHESIS RESULTS
# ============================================================

with open(
    "model/hypothesis_results.pkl",
    "wb"
) as file:

    pickle.dump(
        hypothesis_df,
        file
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("PROJECT ANALYSIS COMPLETED")
print("==============================================")

print(
    f"\nBest Algorithm: {best_model_name}"
)

print(
    f"Best Accuracy: {best_accuracy:.2f}%"
)

print(
    "\nGraphs saved inside the graphs folder."
)

print(
    "\nModels saved inside the model folder."
)

print(
    "\nAll classification algorithms have been compared."
)

print(
    "\nProject analysis completed successfully!"
)