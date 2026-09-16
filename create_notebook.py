import os
import sys
import io
import contextlib
import nbformat as nbf
import pandas as pd
import numpy as np

def generate_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14"
        }
    }

    cells_data = [
        # Cell 1: Notebook Header
        ("markdown", """# Telco Customer Churn Analysis & Data Quality Assessment
### Project: Customer Churn Prediction System
**Phase:** Step 1 — Initial Data Discovery, Structural Audit & Quality Inspection  
**Target:** Predict customer churn (`Churn` feature: `Yes` / `No`)

---

## 1. Project Overview & Objectives
The objective of this notebook is to conduct a thorough exploratory inspection of the Telco Customer Churn dataset prior to any data preprocessing, feature engineering, or model training.

### Key Inspection Goals:
1. Locate and verify the raw dataset file inside the `data/` directory.
2. Ingest the dataset into a pandas DataFrame.
3. Determine dimensionality (row and column counts).
4. Inspect feature column names.
5. Review feature data types and identify type mismatches.
6. Audit missing and null values (both explicit `NaN` and hidden whitespace blanks).
7. Check for duplicate records and evaluate primary key uniqueness.
8. Explore unique values and cardinality across all categorical variables.
9. Examine the distribution and class balance of the target variable (`Churn`).
10. Systematically document data-quality issues and actionable remediation steps.
"""),

        # Cell 2: Step 1 Markdown
        ("markdown", """## Step 1: Locate and Verify Dataset File
Before loading the dataset, verify that the CSV file exists in the relative `data/` folder and inspect its file size.
"""),

        # Cell 3: Step 1 Code
        ("code", """import os
import pandas as pd
import numpy as np

# Define relative path from the notebooks directory to the data directory
data_path = os.path.join('..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Verify file presence and inspect metadata
file_exists = os.path.exists(data_path)
file_size_bytes = os.path.getsize(data_path) if file_exists else 0
file_size_kb = file_size_bytes / 1024

print(f"Dataset path: {data_path}")
print(f"File exists: {file_exists}")
print(f"File size: {file_size_kb:.2f} KB ({file_size_bytes:,} bytes)")
"""),

        # Cell 4: Step 2 Markdown
        ("markdown", """## Step 2: Load Dataset into Python
Load the raw CSV dataset into a pandas DataFrame and preview the first few records.
"""),

        # Cell 5: Step 2 Code
        ("code", """# Load dataset into pandas DataFrame
df = pd.read_csv(data_path)

print("Dataset loaded successfully! Preview of the first 5 records:")
df.head()
"""),

        # Cell 6: Step 3 Markdown
        ("markdown", """## Step 3: Check Number of Rows and Columns
Determine the dataset dimensions (total observations and features).
"""),

        # Cell 7: Step 3 Code
        ("code", """num_rows, num_cols = df.shape
print(f"Total Rows (Customers / Observations): {num_rows:,}")
print(f"Total Columns (Attributes / Features): {num_cols}")
"""),

        # Cell 8: Step 4 Markdown
        ("markdown", """## Step 4: Display Column Names
Examine all column names present in the dataset to understand available features.
"""),

        # Cell 9: Step 4 Code
        ("code", """print(f"All {len(df.columns)} Column Names:")
for i, col in enumerate(df.columns, start=1):
    print(f"  {i:2d}. {col}")
"""),

        # Cell 10: Step 5 Markdown
        ("markdown", """## Step 5: Display Data Types and DataFrame Info
Inspect the inferred data types for each feature and assess memory usage and overall structural schema.
"""),

        # Cell 11: Step 5 Code
        ("code", """print("DataFrame Schema Information:")
print("=" * 60)
df.info()
"""),

        # Cell 12: Step 5 Code (Breakdown)
        ("code", """# Summary count of feature data types
dtype_counts = df.dtypes.value_counts()
print("Data Type Frequency:")
for dtype, count in dtype_counts.items():
    print(f"  - {dtype}: {count} columns")
"""),

        # Cell 13: Step 6 Markdown
        ("markdown", """## Step 6: Check Missing Values
Perform a two-fold missing value inspection:
1. Standard check for explicit `NaN` / `None` values (`df.isnull().sum()`).
2. Deep scan for hidden nulls, such as whitespace strings (`' '`) or empty strings across text/object columns.
"""),

        # Cell 14: Step 6 Code
        ("code", """# 1. Check for standard explicit NaN / null values
explicit_nulls = df.isnull().sum()

# 2. Check for whitespace or empty strings in text/object columns
hidden_whitespace = {}
for col in df.columns:
    if df[col].dtype == 'object' or str(df[col].dtype) == 'str':
        blank_count = (df[col].astype(str).str.strip() == '').sum()
        if blank_count > 0:
            hidden_whitespace[col] = blank_count

null_report = pd.DataFrame({
    'Explicit_NaN_Count': explicit_nulls,
    'Null_Percentage (%)': (explicit_nulls / len(df) * 100).round(2)
})

print("Explicit Missing Values Audit (df.isnull().sum()):")
print(null_report)

print("\\nHidden Missing Values Audit (Whitespace/Blank Strings):")
if hidden_whitespace:
    for col, count in hidden_whitespace.items():
        pct = (count / len(df)) * 100
        print(f"  [ALERT] Column '{col}' contains {count} blank/whitespace strings ({pct:.2f}% of data)!")
else:
    print("  No hidden whitespace values found.")
"""),

        # Cell 15: Step 7 Markdown
        ("markdown", """## Step 7: Check Duplicate Rows
Assess dataset integrity by checking for exact duplicate rows across all features and validating the uniqueness of the primary identifier (`customerID`).
"""),

        # Cell 16: Step 7 Code
        ("code", """# Check for exact duplicate rows across all columns
exact_duplicates = df.duplicated().sum()
print(f"Exact Duplicate Rows Across All Columns: {exact_duplicates}")

# Check uniqueness of primary key (customerID)
unique_customers = df['customerID'].nunique()
print(f"Total Records: {len(df):,}")
print(f"Unique customerID Values: {unique_customers:,}")
print(f"Is customerID strictly unique across all rows? {unique_customers == len(df)}")
"""),

        # Cell 17: Step 8 Markdown
        ("markdown", """## Step 8: Check Unique Values in Categorical Columns
Examine the unique categories, cardinality, and level values across all categorical features (excluding high-cardinality `customerID`).
"""),

        # Cell 18: Step 8 Code
        ("code", """# Identify categorical columns (excluding customerID and TotalCharges)
cat_cols = [c for c in df.columns if (df[c].dtype == 'object' or str(df[c].dtype) == 'str') and c not in ['customerID', 'TotalCharges']]

# Include SeniorCitizen as it represents a binary categorical flag
cat_cols_with_senior = ['SeniorCitizen'] + cat_cols

print(f"Categorical Features Analysis ({len(cat_cols_with_senior)} features):\\n")
for col in cat_cols_with_senior:
    unique_vals = df[col].unique().tolist()
    nunique = df[col].nunique()
    print(f"Feature: '{col}' | Cardinality: {nunique}")
    print(f"  Unique Values: {unique_vals}\\n")
"""),

        # Cell 19: Step 9 Markdown
        ("markdown", """## Step 9: Check the Distribution of the Churn Target Variable
Analyze the target variable (`Churn`), checking class frequencies, class proportions, and the degree of class imbalance.
"""),

        # Cell 20: Step 9 Code
        ("code", """churn_counts = df['Churn'].value_counts(dropna=False)
churn_proportions = df['Churn'].value_counts(normalize=True, dropna=False) * 100

churn_dist_df = pd.DataFrame({
    'Count': churn_counts,
    'Percentage (%)': churn_proportions.round(2)
})

print("Target Variable ('Churn') Distribution:")
print("=" * 45)
print(churn_dist_df)
print("=" * 45)

imbalance_ratio = churn_counts['No'] / churn_counts['Yes']
print(f"\\nClass Ratio (Retained / Churned): {imbalance_ratio:.2f} : 1")
print(f"Churn Prevalence: {churn_proportions['Yes']:.2f}% of customers churned.")
"""),

        # Cell 21: Step 10 Code (Deep Dive into TotalCharges Quality Issue)
        ("markdown", """## Step 10: Identify Data-Quality Issues
Let us investigate the anomalies detected during inspection, specifically the 11 records in `TotalCharges` containing whitespace blanks.
"""),

        # Cell 22: Step 10 Code (TotalCharges Investigation)
        ("code", """# Inspect the rows where TotalCharges is whitespace
blank_tc_rows = df[df['TotalCharges'].astype(str).str.strip() == '']
print(f"Number of rows with blank TotalCharges: {len(blank_tc_rows)}\\n")

print("Key attributes for customers with blank TotalCharges:")
blank_tc_rows[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'Churn']]
"""),

        # Cell 23: Step 10 Code (Verify tenure relationship)
        ("code", """# Verify tenure distribution for customers with blank TotalCharges
print(f"Unique tenure values for customers with blank TotalCharges: {blank_tc_rows['tenure'].unique().tolist()}")
print(f"Unique Churn values for customers with blank TotalCharges: {blank_tc_rows['Churn'].unique().tolist()}")
"""),

        # Cell 24: Step 10 Markdown (Comprehensive Data Quality Findings & Roadmap)
        ("markdown", """## Summary of Data-Quality Issues & Preprocessing Roadmap

Based on the empirical inspection above, we identify the following **5 key data-quality issues** and recommend concrete actions for the subsequent pipeline stages:

| # | Feature / Area | Identified Issue | Root Cause & Impact | Recommended Remediation |
|---|----------------|------------------|---------------------|--------------------------|
| **1** | `TotalCharges` | **Data Type Mismatch & Hidden Nulls** | Stored as `object` (string) instead of `float64` due to 11 rows containing whitespace (`" "`). All 11 records have `tenure = 0` (new accounts with no billed month yet). | Convert column using `pd.to_numeric(df['TotalCharges'], errors='coerce')`. Impute the 11 resulting `NaN` values with `0.0` because `tenure = 0`. |
| **2** | `SeniorCitizen` | **Categorical vs Integer Representation** | Encoded as binary numeric integer (`0` / `1`), while all other demographic flags (`Partner`, `Dependents`) are string `"Yes"` / `"No"`. | Standardize type representation (either map to string `"Yes"`/`"No"` for EDA or keep as binary `0`/`1` for modeling). |
| **3** | `Churn` (Target) | **Class Imbalance (~73.5% vs ~26.5%)** | The majority class (`No`) outnumbers the minority class (`Yes`) by ~2.77 to 1. Standard accuracy will be misleading. | Use stratified train-test splits (`stratify=y`), evaluate using PR-AUC, ROC-AUC, Precision, Recall, F1-Score, and consider class weighting (`scale_pos_weight` / `class_weight='balanced'`) or SMOTE during model training. |
| **4** | Multi-service Categoricals | **Redundant Subcategory Labels** | Features like `MultipleLines` include `'No phone service'`; features like `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies` include `'No internet service'`. | Consolidate or keep as distinct categories depending on encoding strategy. One-hot encoding naturally handles these, or binary consolidation can simplify feature space. |
| **5** | `customerID` | **High-Cardinality Primary Key** | All 7,043 values are unique strings. Passing this column into ML models will cause extreme overfitting or noise. | Drop `customerID` from the feature matrix before exploratory modeling and pipeline training. |

---
### Next Steps for Phase 2:
- Exploratory Data Analysis (EDA) visualizations (univariate, bivariate with `Churn`).
- Data cleaning & type conversion pipeline in `src/`.
- Feature engineering and preprocessing transformations.
- Baseline ML model experimentation in `models/`.
""")
    ]

    exec_env = {}
    
    for cell_type, content in cells_data:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            cell = nbf.v4.new_code_cell(content)
            
            # Execute code cell and capture stdout & last expr
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                try:
                    os.chdir('notebooks')
                    try:
                        lines = content.strip().split('\n')
                        if lines and not lines[-1].startswith(('print', 'import', 'from', '#', 'for', 'if', 'with', 'def', 'while', 'try', 'except')) and '=' not in lines[-1]:
                            exec_code = '\n'.join(lines[:-1])
                            eval_code = lines[-1]
                            if exec_code.strip():
                                exec(exec_code, exec_env)
                            result = eval(eval_code, exec_env)
                            if result is not None:
                                if isinstance(result, pd.DataFrame):
                                    cell.outputs.append(nbf.v4.new_output(
                                        output_type="execute_result",
                                        data={
                                            "text/plain": repr(result),
                                            "text/html": result._repr_html_()
                                        },
                                        execution_count=len(nb.cells)
                                    ))
                                else:
                                    cell.outputs.append(nbf.v4.new_output(
                                        output_type="execute_result",
                                        data={"text/plain": repr(result)},
                                        execution_count=len(nb.cells)
                                    ))
                        else:
                            exec(content, exec_env)
                    finally:
                        os.chdir('..')
                except Exception as e:
                    print(f"Execution error in cell: {e}")
            
            out_text = stdout_capture.getvalue()
            if out_text:
                cell.outputs.insert(0, nbf.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=out_text
                ))
            
            cell.execution_count = len(nb.cells)
            nb.cells.append(cell)

    target_notebook = os.path.join('notebooks', 'customer_churn_analysis.ipynb')
    with open(target_notebook, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook successfully generated at: {target_notebook}")

if __name__ == '__main__':
    generate_notebook()
