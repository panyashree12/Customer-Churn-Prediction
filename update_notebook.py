import os
import sys
import io
import ast
import base64
import contextlib
import nbformat as nbf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def build_full_notebook_with_eda():
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
        # =========================================================================
        # PART 1: DATASET INSPECTION
        # =========================================================================
        ("markdown", """# Telco Customer Churn Analysis & EDA
### Project: Customer Churn Prediction System
**Phase:** 1. Data Discovery & Inspection | 2. Data Cleaning | 3. Exploratory Data Analysis (EDA)  
**Target Variable:** Customer Churn (`Churn`: `Yes` / `No`)

---

## Part 1: Initial Dataset Inspection & Discovery
The first phase focuses on ingesting the raw Telco customer churn dataset, understanding its dimensions, schema, column structure, missing values, duplicates, and class distribution.
"""),

        ("markdown", """### Step 1: Locate and Verify Dataset File
Verify that the raw CSV file exists in the relative `data/` directory and inspect file metadata.
"""),

        ("code", """import os
import pandas as pd
import numpy as np

# Define relative path from notebooks directory to data directory
data_path = os.path.join('..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

file_exists = os.path.exists(data_path)
file_size_bytes = os.path.getsize(data_path) if file_exists else 0
file_size_kb = file_size_bytes / 1024

print(f"Dataset path: {data_path}")
print(f"File exists: {file_exists}")
print(f"File size: {file_size_kb:.2f} KB ({file_size_bytes:,} bytes)")
"""),

        ("markdown", """### Step 2: Load Dataset into Python
Read the CSV into a pandas DataFrame and preview the first 5 records.
"""),

        ("code", """# Load raw dataset into pandas
df = pd.read_csv(data_path)

print("Dataset loaded successfully! First 5 records:")
df.head()
"""),

        ("markdown", """### Step 3: Check Number of Rows and Columns
Examine the dataset dimensions (total customers and features).
"""),

        ("code", """num_rows, num_cols = df.shape
print(f"Total Rows (Customer Observations): {num_rows:,}")
print(f"Total Columns (Features / Attributes): {num_cols}")
"""),

        ("markdown", """### Step 4: Display Column Names
List all 21 feature column names present in the dataset.
"""),

        ("code", """print(f"All {len(df.columns)} Column Names:")
for idx, col in enumerate(df.columns, start=1):
    print(f"  {idx:2d}. {col}")
"""),

        ("markdown", """### Step 5: Display Data Types and DataFrame Info
Inspect the initial inferred data types and memory footprint across all features.
"""),

        ("code", """print("DataFrame Schema & Inferred Data Types:")
print("=" * 65)
df.info()
"""),

        ("code", """# Breakdown of inferred data types
dtype_counts = df.dtypes.value_counts()
print("Inferred Data Type Counts:")
for dtype, count in dtype_counts.items():
    print(f"  - {dtype}: {count} columns")
"""),

        ("markdown", """### Step 6: Check Missing Values
Check for:
1. Standard explicit missing values (`df.isnull().sum()`).
2. Hidden missing values, such as whitespace strings (`' '`) in object/string columns.
"""),

        ("code", """# 1. Explicit null values
explicit_nulls = df.isnull().sum()

# 2. Hidden whitespace values in text columns
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

        ("markdown", """### Step 7: Check Duplicate Rows
Assess dataset integrity by checking for exact duplicate rows across all features and validating the uniqueness of `customerID`.
"""),

        ("code", """# Check exact duplicate rows across all columns
exact_duplicates = df.duplicated().sum()
print(f"Exact Duplicate Rows: {exact_duplicates}")

# Check uniqueness of customerID
unique_customers = df['customerID'].nunique()
print(f"Total Observations: {len(df):,}")
print(f"Unique customerID Values: {unique_customers:,}")
print(f"Is customerID strictly unique per row? {unique_customers == len(df)}")
"""),

        ("markdown", """### Step 8: Check Unique Values in Categorical Columns
Examine the unique categories, cardinality, and level values across all categorical features (excluding high-cardinality `customerID`).
"""),

        ("code", """cat_cols = [c for c in df.columns if (df[c].dtype == 'object' or str(df[c].dtype) == 'str') and c not in ['customerID', 'TotalCharges']]
cat_cols_with_senior = ['SeniorCitizen'] + cat_cols

print(f"Categorical Features Analysis ({len(cat_cols_with_senior)} features):\\n")
for col in cat_cols_with_senior:
    unique_vals = df[col].unique().tolist()
    nunique = df[col].nunique()
    print(f"Feature: '{col}' | Cardinality: {nunique}")
    print(f"  Unique Values: {unique_vals}\\n")
"""),

        ("markdown", """### Step 9: Check the Distribution of the Churn Target Variable
Analyze the target variable (`Churn`), checking class frequencies, proportions, and the degree of class imbalance.
"""),

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

        ("markdown", """### Step 10: Initial Data-Quality Audit
Investigate the 11 records in `TotalCharges` containing whitespace blanks.
"""),

        ("code", """blank_tc_rows = df[df['TotalCharges'].astype(str).str.strip() == '']
print(f"Number of records with blank TotalCharges: {len(blank_tc_rows)}\\n")

print("Key attributes for customers with blank TotalCharges:")
blank_tc_rows[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'Churn']]
"""),

        ("code", """# Verify tenure distribution for customers with blank TotalCharges
print(f"Unique tenure values for blank TotalCharges rows: {blank_tc_rows['tenure'].unique().tolist()}")
print(f"Unique Churn values for blank TotalCharges rows: {blank_tc_rows['Churn'].unique().tolist()}")
"""),

        # =========================================================================
        # PART 2: DATA CLEANING & PREPROCESSING PREPARATION
        # =========================================================================
        ("markdown", """---

## Part 2: Data Cleaning & Preprocessing Preparation
Now that the structural inspection is complete, we perform data cleaning and preprocessing preparation strictly based on justified findings:

1. **Handle Missing Values Appropriately**: Address the 11 whitespace values in `TotalCharges`.
2. **Convert `TotalCharges` to Numeric**: Convert the feature to `float64`.
3. **Verify Duplicate Rows**: Confirm no duplicate customer records exist.
4. **Check Categorical Values for Inconsistencies**: Validate casing, spacing, and cross-feature relationships.
5. **Apply Justified Corrections Only**: Ensure zero arbitrary modifications to valid data.
6. **Show Dataset Shape After Cleaning**.
7. **Show Remaining Missing Values**.
8. **Show Updated Data Types**.
"""),

        ("markdown", """### Step 2.1: Handle Missing Values & Convert `TotalCharges` to Numeric
#### Business & Mathematical Justification:
- In `TotalCharges`, 11 records have blank spaces (`" "`).
- Every single one of these 11 customers has **`tenure = 0`** and **`Churn = 'No'`**.
- Customers with `tenure = 0` are brand-new accounts who signed up in the current billing cycle and have not yet completed a billing month or accumulated any past charges.
- Therefore, setting their accumulated total charges to **`0.0`** is domain-justified and preserves all 7,043 customer records without dropping data.
- After replacing whitespace, convert `TotalCharges` to a numeric (`float64`) feature.
"""),

        ("code", """# Create a working copy for cleaned data
df_clean = df.copy()

# Step 1: Replace whitespace strings with NaN
df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(r'^\\s*$', np.nan, regex=True)

# Verify count of NaN values before imputation
nan_count_before = df_clean['TotalCharges'].isna().sum()
print(f"NaN count in TotalCharges after identifying whitespace: {nan_count_before}")

# Step 2: Convert to numeric float64
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Step 3: Impute NaN with 0.0 based on tenure == 0
df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)

# Verify that no NaN values remain in TotalCharges
nan_count_after = df_clean['TotalCharges'].isna().sum()
print(f"NaN count in TotalCharges after imputation: {nan_count_after}")
print(f"Updated TotalCharges dtype: {df_clean['TotalCharges'].dtype}")
"""),

        ("code", """# Verify the 11 previously blank records after cleaning
cleaned_sample = df_clean.loc[blank_tc_rows.index, ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
print("Cleaned values for the 11 records previously containing whitespace:")
cleaned_sample
"""),

        ("markdown", """### Step 2.2: Check for Duplicate Rows
Confirm whether any duplicate rows exist in the cleaned DataFrame.
"""),

        ("code", """# Verify duplicate rows across the entire dataset
duplicates_clean = df_clean.duplicated().sum()
print(f"Duplicate rows in cleaned dataset: {duplicates_clean}")

# Verify primary key customerID uniqueness
id_unique = df_clean['customerID'].is_unique
print(f"Is customerID strictly unique across all rows? {id_unique}")
"""),

        ("markdown", """### Step 2.3: Check Categorical Values for Inconsistencies
Conduct a rigorous audit of all categorical features:
- Check for unexpected leading or trailing whitespace.
- Check for casing discrepancies (e.g., `'Yes'`, `'yes'`, `'YES'`).
- Verify logical consistency between related features (e.g., `PhoneService` vs `MultipleLines`, `InternetService` vs add-on services).
"""),

        ("code", """# 1. Audit for leading/trailing whitespaces across all text columns
text_cols = df_clean.select_dtypes(include=['object', 'string']).columns

whitespace_issues = {}
for col in text_cols:
    has_whitespace = (df_clean[col].astype(str).str.strip() != df_clean[col].astype(str)).sum()
    if has_whitespace > 0:
        whitespace_issues[col] = has_whitespace

print("Leading/Trailing Whitespace Audit across text columns:")
if whitespace_issues:
    for col, count in whitespace_issues.items():
        print(f"  [ISSUE] {col}: {count} values with whitespace")
else:
    print("  [PASSED] Zero leading or trailing whitespace detected across all text columns.")
"""),

        ("code", """# 2. Audit unique values and casing across all categorical columns
print("Categorical Values & Consistency Check:")
print("=" * 65)
for col in text_cols:
    if col != 'customerID':
        unique_vals = sorted(df_clean[col].unique().tolist())
        print(f"{col:<18}: {unique_vals}")
"""),

        ("code", """# 3. Cross-Feature Consistency Checks

# Check A: MultipleLines vs PhoneService
phone_no_count = (df_clean['PhoneService'] == 'No').sum()
multi_no_phone = (df_clean['MultipleLines'] == 'No phone service').sum()
print(f"Cross-Check A - Phone Service:")
print(f"  PhoneService == 'No': {phone_no_count}")
print(f"  MultipleLines == 'No phone service': {multi_no_phone}")
print(f"  Consistency holds perfectly: {phone_no_count == multi_no_phone}")

# Check B: Internet add-ons vs InternetService
internet_no_count = (df_clean['InternetService'] == 'No').sum()
internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

print(f"\\nCross-Check B - Internet Add-on Services (InternetService == 'No': {internet_no_count}):")
all_consistent = True
for svc in internet_services:
    no_svc_count = (df_clean[svc] == 'No internet service').sum()
    consistent = (no_svc_count == internet_no_count)
    if not consistent:
        all_consistent = False
    print(f"  {svc:<18}: {no_svc_count} 'No internet service' (Consistent: {consistent})")
print(f"  All internet service relationships hold perfectly: {all_consistent}")
"""),

        ("markdown", """### Step 2.4: Justified Data-Quality Corrections Summary
Every data cleaning action taken is mathematically and contextually justified:
1. **`TotalCharges` Conversion & Imputation**:
   - Converted from `object` to `float64`.
   - The 11 whitespace values were imputed as `0.0` because their `tenure` is `0`.
2. **Feature Integrity Preservation**:
   - Categorical values are consistent with no typos, no irregular casings, and no extra spaces.
   - Logical relationships between parent services (`PhoneService`, `InternetService`) and child options (`MultipleLines`, add-on features) are 100% consistent across all 7,043 rows.
3. **`customerID` & Modeling Preparation**:
   - `customerID` is confirmed to be 100% unique across all rows. It will be excluded from feature vectors during model training to avoid data leakage.
"""),

        ("markdown", """### Step 2.5: Dataset Verification Post-Cleaning
We now inspect the final state of the cleaned dataset across shape, remaining nulls, and updated data types.
"""),

        ("code", """# 1. Dataset Shape After Cleaning
clean_rows, clean_cols = df_clean.shape
print("1. Dataset Dimensions Post-Cleaning:")
print(f"   Total Rows: {clean_rows:,} (Original: {df.shape[0]:,})")
print(f"   Total Columns: {clean_cols} (Original: {df.shape[1]})")
"""),

        ("code", """# 2. Remaining Missing Values After Cleaning
remaining_nulls = df_clean.isnull().sum()
total_remaining_nulls = remaining_nulls.sum()

print("2. Remaining Missing Values Audit:")
print(f"   Total Missing Values across all columns: {total_remaining_nulls}")
print("-" * 50)
print(remaining_nulls[remaining_nulls > 0] if total_remaining_nulls > 0 else "All 21 columns have 0 missing values.")
"""),

        ("code", """# 3. Updated Data Types After Cleaning
print("3. Updated Schema & Data Types:")
print("=" * 65)
df_clean.info()
"""),

        ("code", """# Compare data types before and after cleaning
dtype_comparison = pd.DataFrame({
    'Before_Cleaning': df.dtypes,
    'After_Cleaning': df_clean.dtypes
})
dtype_comparison['Type_Changed'] = dtype_comparison['Before_Cleaning'] != dtype_comparison['After_Cleaning']

print("Data Types Comparison:")
print(dtype_comparison)
"""),

        ("markdown", """### Step 2.6: Statistical Summary of Cleaned Numerical Features
Inspect descriptive statistics for all numeric features now that `TotalCharges` has been converted to `float64`.
"""),

        ("code", """# Statistical summary of numerical columns
print("Summary Statistics for Numerical Features:")
df_clean[['tenure', 'MonthlyCharges', 'TotalCharges']].describe().round(2)
"""),

        # =========================================================================
        # PART 3: EXPLORATORY DATA ANALYSIS (EDA)
        # =========================================================================
        ("markdown", """---

## Exploratory Data Analysis (EDA)

In this section, we conduct a comprehensive exploratory investigation to uncover how key customer attributes, subscription plans, service add-ons, and billing characteristics relate to **Customer Churn (`Churn`)**.

We analyze the relationship between `Churn` and the following **12 key variables**:
1. **`Contract`**: Contract commitment type (Month-to-month, One year, Two year).
2. **`tenure`**: Duration of customer subscription in months.
3. **`MonthlyCharges`**: Current monthly recurring charge ($).
4. **`TotalCharges`**: Total cumulative billings charged to the customer ($).
5. **`PaymentMethod`**: Billing payment channel.
6. **`InternetService`**: Internet connection type (Fiber optic, DSL, None).
7. **`TechSupport`**: Technical assistance add-on subscription.
8. **`OnlineSecurity`**: Cyber-security add-on subscription.
9. **`PaperlessBilling`**: Paperless e-billing vs. mailed paper invoice.
10. **`SeniorCitizen`**: Senior demographic flag (65+ years old).
11. **`Partner`**: Relationship status flag.
12. **`Dependents`**: Family dependents status flag.

Visualizations include count plots, bar charts with percentage labels, histograms/KDE distribution plots, box plots, and a correlation heatmap. Short analytical explanations accompany each visualization to detail empirical findings.
"""),

        ("markdown", """### EDA Setup & Visualization Utilities
Configure visualization aesthetics using `matplotlib` and `seaborn`.
"""),

        ("code", """import matplotlib.pyplot as plt
import seaborn as sns

# Configure aesthetic style for publication-quality figures
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 110

# Define consistent color palette
palette_churn = {'No': '#2b5c8f', 'Yes': '#d95f02'}
"""),

        ("markdown", """### Categorical Churn Rates Summary Table
Before individual deep dives, we compute the exact cross-tabulations and **churn percentages** across all key categorical variables.
"""),

        ("code", """# Calculate Churn Rates for All Key Categorical Features
cat_features = [
    'Contract', 'PaymentMethod', 'InternetService', 
    'TechSupport', 'OnlineSecurity', 'PaperlessBilling', 
    'SeniorCitizen', 'Partner', 'Dependents'
]

churn_summary_list = []
for feat in cat_features:
    ct = pd.crosstab(df_clean[feat], df_clean['Churn'])
    pct = pd.crosstab(df_clean[feat], df_clean['Churn'], normalize='index') * 100
    
    summary_df = pd.DataFrame({
        'Feature': feat,
        'Category': ct.index.astype(str),
        'Total_Customers': ct['No'] + ct['Yes'],
        'Retained_No': ct['No'].values,
        'Churned_Yes': ct['Yes'].values,
        'Churn_Rate (%)': pct['Yes'].round(2).values
    })
    churn_summary_list.append(summary_df)

full_churn_rates = pd.concat(churn_summary_list, ignore_index=True)
print("Consolidated Churn Rates Across Key Categorical Features:")
full_churn_rates
"""),

        # -------------------------------------------------------------------------
        # 1. Contract vs Churn
        # -------------------------------------------------------------------------
        ("markdown", """### 1. Contract Type vs. Churn
Investigate how contract commitment length (Month-to-month, One year, Two year) influences churn propensity.
"""),

        ("code", """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1A: Count of Retained vs Churned by Contract
sns.countplot(data=df_clean, x='Contract', hue='Churn', palette=palette_churn, ax=axes[0])
axes[0].set_title('Customer Count by Contract Type & Churn', fontsize=13, weight='bold', pad=10)
axes[0].set_xlabel('Contract Type', fontsize=11)
axes[0].set_ylabel('Number of Customers', fontsize=11)
axes[0].legend(title='Churn', frameon=True)

for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].annotate(f'{int(height):,}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                         textcoords='offset points')

# Plot 1B: Churn Percentage by Contract
contract_pct = df_clean.groupby('Contract')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
contract_order = ['Month-to-month', 'One year', 'Two year']
sns.barplot(data=contract_pct, x='Contract', y='Churn_Rate_%', order=contract_order, palette='Reds_r', ax=axes[1])
axes[1].set_title('Churn Rate (%) by Contract Type', fontsize=13, weight='bold', pad=10)
axes[1].set_xlabel('Contract Type', fontsize=11)
axes[1].set_ylabel('Churn Rate (%)', fontsize=11)
axes[1].set_ylim(0, 55)

for p in axes[1].patches:
    height = p.get_height()
    axes[1].annotate(f'{height:.2f}%',
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=10, weight='bold', xytext=(0, 4),
                     textcoords='offset points')

plt.tight_layout()
plt.show()
"""),

        ("markdown", """#### Empirical Pattern & Business Takeaway: Contract Type
- **Month-to-month** contracts exhibit an alarming **42.71% churn rate** (1,655 out of 3,875 customers). Over **88.5% of all churned customers** in the entire dataset are on month-to-month plans.
- **One-year** contracts drop churn down to **11.27%** (166 churned).
- **Two-year** contracts experience an exceptionally low churn rate of just **2.83%** (only 48 churned out of 1,695).
- **Key Insight**: Contract duration is one of the strongest protective barriers against churn. Customers without contractual commitment face zero exit frictions. Incentivizing annual or multi-year commitments is the most direct retention lever.
"""),

        # -------------------------------------------------------------------------
        # 2. Tenure vs Churn
        # -------------------------------------------------------------------------
        ("markdown", """### 2. Tenure vs. Churn
Examine the relationship between customer subscription longevity (`tenure` in months) and churn likelihood.
"""),

        ("code", """fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 2A: Histogram & KDE of Tenure by Churn Status
sns.histplot(data=df_clean, x='tenure', hue='Churn', palette=palette_churn, 
             kde=True, element='step', common_norm=False, bins=36, ax=axes[0])
axes[0].set_title('Tenure Distribution by Churn Status', fontsize=13, weight='bold', pad=10)
axes[0].set_xlabel('Tenure (Months)', fontsize=11)
axes[0].set_ylabel('Customer Count', fontsize=11)

# Plot 2B: Box Plot of Tenure by Churn
sns.boxplot(data=df_clean, x='Churn', y='tenure', palette=palette_churn, ax=axes[1], width=0.45)
axes[1].set_title('Tenure Box Plot (Median & IQR)', fontsize=13, weight='bold', pad=10)
axes[1].set_xlabel('Churn Status', fontsize=11)
axes[1].set_ylabel('Tenure (Months)', fontsize=11)

# Annotate Medians on Box Plot
medians = df_clean.groupby('Churn')['tenure'].median()
for i, (churn_val, median_val) in enumerate(medians.items()):
    axes[1].text(i, median_val + 2, f'Median: {median_val:.0f} mos', 
                 horizontalalignment='center', fontsize=10, weight='bold', color='black')

plt.tight_layout()
plt.show()
"""),

        ("markdown", """#### Empirical Pattern & Business Takeaway: Tenure
- **Extreme Front-Loaded Churn Risk**: The churn distribution exhibits a massive peak in the **first 1 to 5 months**.
- **Median Tenure Comparison**:
  - Churned customers have a median tenure of only **10.0 months** (mean: 17.98 months, 25th percentile: 2.0 months).
  - Retained customers have a median tenure of **38.0 months** (mean: 37.57 months).
- **Key Insight**: Customer churn is heavily concentrated in the onboarding phase. If a customer remains with the company past 24 months, churn risk diminishes dramatically. Retention programs must intervene during the initial 90-day window.
"""),

        # -------------------------------------------------------------------------
        # 3 & 4. MonthlyCharges & TotalCharges vs Churn
        # -------------------------------------------------------------------------
        ("markdown", """### 3 & 4. MonthlyCharges and TotalCharges vs. Churn
Analyze how recurring monthly cost (`MonthlyCharges`) and cumulative lifetime billing (`TotalCharges`) correlate with churn decisions.
"""),

        ("code", """fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 3A: MonthlyCharges KDE Distribution
sns.kdeplot(data=df_clean, x='MonthlyCharges', hue='Churn', palette=palette_churn, fill=True, common_norm=False, ax=axes[0, 0])
axes[0, 0].set_title('Monthly Charges Distribution (KDE)', fontsize=12, weight='bold')
axes[0, 0].set_xlabel('Monthly Charges ($)', fontsize=10)
axes[0, 0].set_ylabel('Density', fontsize=10)

# Subplot 3B: MonthlyCharges Box Plot
sns.boxplot(data=df_clean, x='Churn', y='MonthlyCharges', palette=palette_churn, width=0.45, ax=axes[0, 1])
axes[0, 1].set_title('Monthly Charges Box Plot', fontsize=12, weight='bold')
axes[0, 1].set_xlabel('Churn Status', fontsize=10)
axes[0, 1].set_ylabel('Monthly Charges ($)', fontsize=10)
mc_medians = df_clean.groupby('Churn')['MonthlyCharges'].median()
for i, (status, val) in enumerate(mc_medians.items()):
    axes[0, 1].text(i, val + 2, f'Median: ${val:.2f}', ha='center', weight='bold')

# Subplot 4A: TotalCharges KDE Distribution
sns.kdeplot(data=df_clean, x='TotalCharges', hue='Churn', palette=palette_churn, fill=True, common_norm=False, ax=axes[1, 0])
axes[1, 0].set_title('Total Charges Distribution (KDE)', fontsize=12, weight='bold')
axes[1, 0].set_xlabel('Total Charges ($)', fontsize=10)
axes[1, 0].set_ylabel('Density', fontsize=10)

# Subplot 4B: TotalCharges Box Plot
sns.boxplot(data=df_clean, x='Churn', y='TotalCharges', palette=palette_churn, width=0.45, ax=axes[1, 1])
axes[1, 1].set_title('Total Charges Box Plot', fontsize=12, weight='bold')
axes[1, 1].set_xlabel('Churn Status', fontsize=10)
axes[1, 1].set_ylabel('Total Charges ($)', fontsize=10)
tc_medians = df_clean.groupby('Churn')['TotalCharges'].median()
for i, (status, val) in enumerate(tc_medians.items()):
    axes[1, 1].text(i, val + 200, f'Median: ${val:.2f}', ha='center', weight='bold')

plt.tight_layout()
plt.show()
"""),

        ("markdown", """#### Empirical Pattern & Business Takeaway: Charges
- **MonthlyCharges**:
  - Churned customers pay significantly higher monthly rates: median **\$79.65** (mean: \$74.44) compared to retained customers with median **\$64.43** (mean: \$61.27).
  - The KDE plot demonstrates that churn density surges dramatically in the **\$70 - \$105 per month** range. High pricing is a primary churn catalyst.
- **TotalCharges**:
  - Retained customers have a much higher median total spend (**\$1,679.52**) than churners (**\$703.55**).
  - While churners pay more per month, their lifetime billing is substantially curtailed due to rapid cancellation.
"""),

        # -------------------------------------------------------------------------
        # 5 & 9. PaymentMethod & PaperlessBilling vs Churn
        # -------------------------------------------------------------------------
        ("markdown", """### 5 & 9. PaymentMethod and PaperlessBilling vs. Churn
Assess how billing delivery (`PaperlessBilling`) and payment channels (`PaymentMethod`) relate to customer attrition.
"""),

        ("code", """fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 5A: Churn Rate by PaymentMethod
pm_pct = df_clean.groupby('PaymentMethod')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
pm_pct = pm_pct.sort_values(by='Churn_Rate_%', ascending=False)

sns.barplot(data=pm_pct, y='PaymentMethod', x='Churn_Rate_%', palette='Oranges_r', ax=axes[0])
axes[0].set_title('Churn Rate (%) by Payment Method', fontsize=13, weight='bold', pad=10)
axes[0].set_xlabel('Churn Rate (%)', fontsize=11)
axes[0].set_ylabel('Payment Method', fontsize=11)
axes[0].set_xlim(0, 55)

for p in axes[0].patches:
    width = p.get_width()
    axes[0].annotate(f'{width:.2f}%',
                     (width + 1, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', fontsize=10, weight='bold')

# Plot 5B: Churn Rate by PaperlessBilling
pb_pct = df_clean.groupby('PaperlessBilling')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
sns.barplot(data=pb_pct, x='PaperlessBilling', y='Churn_Rate_%', palette='Purples_r', ax=axes[1], width=0.45)
axes[1].set_title('Churn Rate (%) by Paperless Billing', fontsize=13, weight='bold', pad=10)
axes[1].set_xlabel('Paperless Billing Enrolled', fontsize=11)
axes[1].set_ylabel('Churn Rate (%)', fontsize=11)
axes[1].set_ylim(0, 42)

for p in axes[1].patches:
    height = p.get_height()
    axes[1].annotate(f'{height:.2f}%',
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=10, weight='bold', xytext=(0, 4),
                     textcoords='offset points')

plt.tight_layout()
plt.show()
"""),

        ("markdown", """#### Empirical Pattern & Business Takeaway: Payment & Billing Channels
- **Electronic Check Disproportionate Risk**:
  - Customers paying via **Electronic check** experience an extraordinarily high churn rate of **45.29%** (1,071 churners out of 2,365 users).
  - In comparison, customers using automated recurring payment options (**Credit card (automatic)**: **15.24%**, **Bank transfer (automatic)**: **16.71%**) and **Mailed check** (**19.11%**) churn at nearly **one-third the rate**.
- **Paperless Billing**:
  - Customers enrolled in **Paperless billing** churn at **33.57%**, compared to only **16.33%** for paper billing recipients.
- **Key Insight**: Electronic check is a manual, high-friction payment method where bill shock is actively confronted every month. Automated autopay models lock in recurring retention.
"""),

        # -------------------------------------------------------------------------
        # 6, 7 & 8. InternetService, TechSupport & OnlineSecurity vs Churn
        # -------------------------------------------------------------------------
        ("markdown", """### 6, 7 & 8. InternetService, TechSupport, and OnlineSecurity vs. Churn
Investigate core internet connectivity tier and essential support/security add-ons.
"""),

        ("code", """fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 6A: InternetService Churn Rate
is_pct = df_clean.groupby('InternetService')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
sns.barplot(data=is_pct, x='InternetService', y='Churn_Rate_%', palette='Blues_r', ax=axes[0])
axes[0].set_title('Churn Rate by Internet Service', fontsize=12, weight='bold')
axes[0].set_xlabel('Internet Service Tier', fontsize=10)
axes[0].set_ylabel('Churn Rate (%)', fontsize=10)
axes[0].set_ylim(0, 50)
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', weight='bold', xytext=(0, 3), textcoords='offset points')

# Plot 6B: TechSupport Churn Rate
ts_pct = df_clean.groupby('TechSupport')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
sns.barplot(data=ts_pct, x='TechSupport', y='Churn_Rate_%', palette='Greens_r', ax=axes[1])
axes[1].set_title('Churn Rate by Tech Support Add-On', fontsize=12, weight='bold')
axes[1].set_xlabel('Tech Support Status', fontsize=10)
axes[1].set_ylabel('Churn Rate (%)', fontsize=10)
axes[1].set_ylim(0, 50)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', weight='bold', xytext=(0, 3), textcoords='offset points')

# Plot 6C: OnlineSecurity Churn Rate
os_pct = df_clean.groupby('OnlineSecurity')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
sns.barplot(data=os_pct, x='OnlineSecurity', y='Churn_Rate_%', palette='Teal_r', ax=axes[2])
axes[2].set_title('Churn Rate by Online Security Add-On', fontsize=12, weight='bold')
axes[2].set_xlabel('Online Security Status', fontsize=10)
axes[2].set_ylabel('Churn Rate (%)', fontsize=10)
axes[2].set_ylim(0, 50)
for p in axes[2].patches:
    axes[2].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', weight='bold', xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.show()
"""),

        ("markdown", """#### Empirical Pattern & Business Takeaway: Internet & Security Services
- **Fiber Optic Attrition**:
  - **Fiber optic** users experience a massive **41.89% churn rate** (1,297 churners out of 3,096 customers).
  - In contrast, **DSL** customers churn at **18.96%**, and customers with **No internet service** churn at only **7.40%**.
  - Fiber optic is the most expensive tier, suggesting service quality, expectation mismatches, or aggressive competitor fiber offerings drive cancellations.
- **Protective "Sticky" Services**:
  - Customers with **TechSupport = 'Yes'** churn at only **15.17%**, compared to **41.64%** for those without it (a **2.7x reduction**).
  - Customers with **OnlineSecurity = 'Yes'** churn at only **14.61%**, compared to **41.77%** for those without it (a **2.8x reduction**).
  - **Key Insight**: Security and support features function as high-retention "stickiness" mechanisms. Bundling free or discounted tech support with fiber optic plans could drastically stem subscriber loss.
"""),

        # -------------------------------------------------------------------------
        # 10, 11 & 12. Demographic Factors vs Churn
        # -------------------------------------------------------------------------
        ("markdown", """### 10, 11 & 12. Demographic Factors (`SeniorCitizen`, `Partner`, `Dependents`) vs. Churn
Explore demographic influences: age brackets and household family commitments.
"""),

        ("code", """fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Plot 10A: SeniorCitizen Churn Rate
senior_pct = df_clean.groupby('SeniorCitizen')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
senior_pct['Senior_Label'] = senior_pct['SeniorCitizen'].map({0: 'Non-Senior (0)', 1: 'Senior Citizen (1)'})
sns.barplot(data=senior_pct, x='Senior_Label', y='Churn_Rate_%', palette='coolwarm', ax=axes[0], width=0.45)
axes[0].set_title('Churn Rate: Senior Citizens', fontsize=12, weight='bold')
axes[0].set_xlabel('Senior Citizen Status', fontsize=10)
axes[0].set_ylabel('Churn Rate (%)', fontsize=10)
axes[0].set_ylim(0, 50)
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', weight='bold', xytext=(0, 3), textcoords='offset points')

# Plot 10B: Partner Churn Rate
partner_pct = df_clean.groupby('Partner')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
sns.barplot(data=partner_pct, x='Partner', y='Churn_Rate_%', palette='Blues_r', ax=axes[1], width=0.45)
axes[1].set_title('Churn Rate: Has Partner', fontsize=12, weight='bold')
axes[1].set_xlabel('Partner Status', fontsize=10)
axes[1].set_ylabel('Churn Rate (%)', fontsize=10)
axes[1].set_ylim(0, 40)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', weight='bold', xytext=(0, 3), textcoords='offset points')

# Plot 10C: Dependents Churn Rate
dep_pct = df_clean.groupby('Dependents')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='Churn_Rate_%')
sns.barplot(data=dep_pct, x='Dependents', y='Churn_Rate_%', palette='Greens_r', ax=axes[2], width=0.45)
axes[2].set_title('Churn Rate: Has Dependents', fontsize=12, weight='bold')
axes[2].set_xlabel('Dependents Status', fontsize=10)
axes[2].set_ylabel('Churn Rate (%)', fontsize=10)
axes[2].set_ylim(0, 40)
for p in axes[2].patches:
    axes[2].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', weight='bold', xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.show()
"""),

        ("markdown", """#### Empirical Pattern & Business Takeaway: Demographics
- **Senior Citizens**:
  - **Senior citizens** exhibit a significantly higher churn rate of **41.68%** (476 out of 1,142), compared to **23.61%** for non-seniors. Fixed incomes and tech complexity may contribute to higher churn.
- **Family Ties & Household Stability**:
  - Customers without a partner churn at **32.96%**, whereas customers with a partner churn at **19.66%**.
  - Customers without dependents churn at **31.28%**, compared to only **15.45%** for customers with dependents (a **50% lower churn rate**).
  - **Key Insight**: Multi-person households and families demonstrate greater inertia and higher switching costs, making them highly stable long-term accounts.
"""),

        # -------------------------------------------------------------------------
        # Correlation Heatmap
        # -------------------------------------------------------------------------
        ("markdown", """### Correlation Heatmap of Numeric & Binary Features
Evaluate linear correlations between continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`), demographic indicator `SeniorCitizen`, and the target variable `Churn_Numeric` (encoded as `Yes`=1, `No`=0).
"""),

        ("code", """# Create numeric target encoding for correlation analysis
df_clean['Churn_Numeric'] = (df_clean['Churn'] == 'Yes').astype(int)

# Select relevant numeric features
corr_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_Numeric']
corr_matrix = df_clean[corr_features].corr()

# Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1, 
            square=True, linewidths=1, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Correlation Heatmap of Key Numerical Features & Churn', fontsize=13, weight='bold', pad=12)
plt.tight_layout()
plt.show()
"""),

        ("markdown", """#### Empirical Findings: Correlation Analysis
1. **Tenure & Churn (`r = -0.352`)**:
   - Strongest negative linear relationship with churn. As tenure increases, churn probability drops steadily.
2. **MonthlyCharges & Churn (`r = +0.193`)**:
   - Positive correlation. Higher monthly fees increase the likelihood of customer departure.
3. **TotalCharges & Churn (`r = -0.198`)**:
   - Negative correlation. Customers with large lifetime cumulative billings are established, loyal subscribers who churn less often.
4. **Tenure & TotalCharges Collinearity (`r = +0.826`)**:
   - Very high positive correlation between tenure and total charges. In linear/logistic models, this multicollinearity should be monitored through regularization (L1/L2) or feature selection.
5. **SeniorCitizen & Churn (`r = +0.151`)**:
   - Modest positive correlation confirming senior citizens churn more frequently.
"""),

        # -------------------------------------------------------------------------
        # EDA Summary & Conclusions
        # -------------------------------------------------------------------------
        ("markdown", """### Key Analytical Findings & Feature Engineering Roadmap

| Rank | Feature / Driver | Empirical Observation | Modeling Recommendation |
|---|---|---|---|
| **1** | **`Contract`** | Month-to-month contracts churn at **42.71%**, compared to **2.83%** for 2-year plans (>15x difference). | Essential categorical feature; one-hot encode or target encode. |
| **2** | **`tenure`** | Median tenure is **10 months** for churners vs. **38 months** for loyal customers. Churn peaks in months 1–5. | Create binned tenure cohorts (e.g., 0-12m, 13-24m, 25-48m, 49-72m). |
| **3** | **`PaymentMethod`** | **Electronic check** churn is **45.29%**, vs. ~15-17% for automated credit card / bank draft. | Critical predictor; flag automated vs. manual payment methods. |
| **4** | **`InternetService`** | **Fiber optic** churn is **41.89%**, while DSL is **18.96%** and no internet is **7.40%**. | High-importance feature; interaction with MonthlyCharges. |
| **5** | **`TechSupport` & `OnlineSecurity`** | Lack of support/security spikes churn to **>41%**; having both reduces churn to **~15%**. | Create a composite "Service Protection Index" feature. |
| **6** | **`MonthlyCharges`** | Churned median is **\$79.65** vs. **\$64.43** for non-churned. Heavy churn cluster at \$70-\$100. | Scale numerically (StandardScaler/RobustScaler). |
| **7** | **`Dependents` & `Partner`** | Family commitments cut churn in half (has dependents: **15.45%** vs. none: **31.28%**). | Create a household stability binary flag. |

The exploratory analysis confirms strong, statistically significant signals across contract commitments, tenure lifecycle, service tiers, and pricing. The dataset is thoroughly prepared for feature engineering and machine learning model training in the next phase.
""")
    ]

    exec_env = {}
    
    for idx, (cell_type, content) in enumerate(cells_data):
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            cell = nbf.v4.new_code_cell(content)
            
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                try:
                    os.chdir('notebooks')
                    try:
                        # Parse AST to check for expression vs statements
                        tree = ast.parse(content)
                        result = None
                        if tree.body and isinstance(tree.body[-1], ast.Expr):
                            last_expr = tree.body.pop()
                            if tree.body:
                                exec(compile(ast.Module(body=tree.body, type_ignores=[]), filename="<ast>", mode="exec"), exec_env)
                            result = eval(compile(ast.Expression(body=last_expr.value), filename="<ast>", mode="eval"), exec_env)
                        else:
                            exec(content, exec_env)
                            
                        # If a matplotlib figure was generated (plt.get_fignums()), capture it as base64 png
                        fig_nums = plt.get_fignums()
                        if fig_nums:
                            for fig_num in fig_nums:
                                fig = plt.figure(fig_num)
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                                buf.seek(0)
                                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                                cell.outputs.append(nbf.v4.new_output(
                                    output_type="display_data",
                                    data={
                                        "image/png": img_b64,
                                        "text/plain": f"<Figure size {fig.get_size_inches()[0]*100}x{fig.get_size_inches()[1]*100}>"
                                    }
                                ))
                            plt.close('all')
                            
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
                            elif isinstance(result, pd.Series):
                                cell.outputs.append(nbf.v4.new_output(
                                    output_type="execute_result",
                                    data={
                                        "text/plain": repr(result),
                                        "text/html": result.to_frame()._repr_html_()
                                    },
                                    execution_count=len(nb.cells)
                                ))
                            else:
                                cell.outputs.append(nbf.v4.new_output(
                                    output_type="execute_result",
                                    data={"text/plain": repr(result)},
                                    execution_count=len(nb.cells)
                                ))
                    finally:
                        os.chdir('..')
                except Exception as e:
                    print(f"Execution error in cell {idx}: {e}")
                    raise e
            
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
    
    print(f"Notebook with EDA successfully generated and saved at: {target_notebook}")

if __name__ == '__main__':
    build_full_notebook_with_eda()
