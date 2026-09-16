import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. CLEAN DATA
# ============================================================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove duplicates
df = df.drop_duplicates()

# Remove rows where target is missing
df = df.dropna(subset=["Churn"])


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

# Create monthly charge groups
df["MonthlyChargesGroup"] = pd.cut(
    df["MonthlyCharges"],
    bins=[-1, 30, 60, 90, float("inf")],
    labels=["Low", "Medium", "High", "Very High"]
)

# Count subscribed services
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

for col in service_columns:
    if col in df.columns:
        df[col + "_binary"] = (
            df[col].astype(str).str.lower().isin(
                ["yes", "yes internet service"]
            ).astype(int)
        )

binary_service_columns = [
    col + "_binary"
    for col in service_columns
    if col in df.columns
]

df["TotalServicesUsed"] = df[binary_service_columns].sum(axis=1)


# ============================================================
# 4. PREPARE X AND y
# ============================================================

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# CustomerID is an identifier, not a useful predictive feature
drop_columns = ["Churn", "customerID"]

X = df.drop(
    columns=[col for col in drop_columns if col in df.columns]
)


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("\nCategorical features:", categorical_features)


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# 9. TRAIN AND EVALUATE
# ============================================================

results = []
trained_pipelines = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_pipelines[model_name] = pipeline

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print(results_df.to_string(index=False))


# Select best model based primarily on ROC-AUC
best_model_name = results_df.loc[
    results_df["ROC-AUC"].idxmax(),
    "Model"
]

best_pipeline = trained_pipelines[best_model_name]

print("\nBest Model:", best_model_name)


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_pipeline,
    "models/churn_model.pkl"
)

print("\nSaved model:")
print("models/churn_model.pkl")


# ============================================================
# 12. SAVE MODEL RESULTS
# ============================================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("Saved:")
print("models/model_comparison.csv")


# ============================================================
# 13. CHURN RISK PREDICTION
# ============================================================

test_results = X_test.copy()

test_results["ActualChurn"] = y_test.values

test_results["ChurnProbability"] = (
    best_pipeline.predict_proba(X_test)[:, 1]
)

def risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"

test_results["RiskCategory"] = (
    test_results["ChurnProbability"]
    .apply(risk_category)
)

test_results = test_results.sort_values(
    "ChurnProbability",
    ascending=False
)

print("\nTop High-Risk Customers:")
print(
    test_results[
        ["ChurnProbability", "RiskCategory"]
    ].head(20)
)


# ============================================================
# 14. SAVE RISK RESULTS
# ============================================================

test_results.to_csv(
    "models/customer_risk_predictions.csv"
)

print("\nSaved:")
print("models/customer_risk_predictions.csv")

print("\nPROJECT TRAINING COMPLETED SUCCESSFULLY.")