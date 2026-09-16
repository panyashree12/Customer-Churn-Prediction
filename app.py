import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "churn_model.pkl"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_comparison.csv"
)

RISK_PATH = os.path.join(
    BASE_DIR,
    "models",
    "customer_risk_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_risk_data():
    return pd.read_csv(RISK_PATH)


try:
    df = load_data()
    model = load_model()
    results_df = load_results()
    risk_df = load_risk_data()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction & Retention Analytics")

st.markdown(
    """
    **Machine Learning System for identifying customers who are likely to churn
    and supporting proactive customer-retention decisions.**
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview",
        "Churn Analysis",
        "Customer Risk Analysis",
        "Model Performance",
        "Individual Prediction",
        "Business Recommendations"
    ]
)


# ============================================================
# PREPARE BASIC DATA
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["Churn"] = df["Churn"].astype(str)

total_customers = len(df)

churned_customers = (
    df["Churn"].str.lower() == "yes"
).sum()

churn_rate = (
    churned_customers / total_customers * 100
)

high_risk_customers = (
    risk_df["RiskCategory"] == "High Risk"
).sum()


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Overall Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "High-Risk Customers",
        f"{high_risk_customers:,}"
    )

    st.divider()

    st.subheader("Project Overview")

    st.write(
        """
        This project analyzes customer behavior and builds machine-learning
        models to predict customer churn.

        The system compares Logistic Regression, Random Forest and XGBoost,
        identifies important churn-related features, estimates customer churn
        probability and categorizes customers into different risk levels.
        """
    )

    st.subheader("Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        churn_counts.index,
        churn_counts.values
    )

    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Churn Distribution")

    st.pyplot(fig)


# ============================================================
# PAGE 2 — CHURN ANALYSIS
# ============================================================

elif page == "Churn Analysis":

    st.header("Customer Churn Analysis")

    st.subheader("Churn by Contract")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    if "Yes" in contract_churn.columns:
        st.bar_chart(contract_churn["Yes"])

    st.dataframe(
        contract_churn.round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("Churn by Payment Method")

    payment_churn = pd.crosstab(
        df["PaymentMethod"],
        df["Churn"],
        normalize="index"
    ) * 100

    if "Yes" in payment_churn.columns:
        st.bar_chart(payment_churn["Yes"])

    st.dataframe(
        payment_churn.round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("Monthly Charges vs Churn")

    fig, ax = plt.subplots()

    df.boxplot(
        column="MonthlyCharges",
        by="Churn",
        ax=ax
    )

    ax.set_title("Monthly Charges by Churn Status")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")

    plt.suptitle("")

    st.pyplot(fig)

    st.divider()

    st.subheader("Tenure Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df.loc[df["Churn"] == "No", "tenure"],
        bins=20,
        alpha=0.7,
        label="No Churn"
    )

    ax.hist(
        df.loc[df["Churn"] == "Yes", "tenure"],
        bins=20,
        alpha=0.7,
        label="Churn"
    )

    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Tenure Distribution by Churn")
    ax.legend()

    st.pyplot(fig)


# ============================================================
# PAGE 3 — CUSTOMER RISK ANALYSIS
# ============================================================

elif page == "Customer Risk Analysis":

    st.header("Customer Churn Risk Analysis")

    st.write(
        "Customers are ranked according to their predicted churn probability."
    )

    col1, col2, col3 = st.columns(3)

    low_risk = (
        risk_df["RiskCategory"] == "Low Risk"
    ).sum()

    medium_risk = (
        risk_df["RiskCategory"] == "Medium Risk"
    ).sum()

    high_risk = (
        risk_df["RiskCategory"] == "High Risk"
    ).sum()

    col1.metric("Low Risk", low_risk)
    col2.metric("Medium Risk", medium_risk)
    col3.metric("High Risk", high_risk)

    st.divider()

    st.subheader("Risk Distribution")

    risk_counts = risk_df["RiskCategory"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        risk_counts.index,
        risk_counts.values
    )

    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Churn Risk Distribution")

    st.pyplot(fig)

    st.divider()

    st.subheader("Highest-Risk Customers")

    display_columns = [
        col for col in [
            "customerID",
            "ChurnProbability",
            "RiskCategory"
        ]
        if col in risk_df.columns
    ]

    high_risk_table = risk_df[
        risk_df["RiskCategory"] == "High Risk"
    ][display_columns].head(50)

    if "ChurnProbability" in high_risk_table.columns:
        high_risk_table = high_risk_table.copy()

        high_risk_table["ChurnProbability"] = (
            high_risk_table["ChurnProbability"] * 100
        ).round(2)

        high_risk_table = high_risk_table.rename(
            columns={
                "ChurnProbability": "Churn Probability (%)"
            }
        )

    st.dataframe(
        high_risk_table,
        use_container_width=True
    )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    st.write(
        "Three classification models were evaluated using the same test dataset."
    )

    # Format metrics
    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]

    formatted_results = results_df.copy()

    for col in metric_columns:
        if col in formatted_results.columns:
            formatted_results[col] = (
                formatted_results[col] * 100
            ).round(2)

    st.subheader("Model Comparison")

    st.dataframe(
        formatted_results,
        use_container_width=True
    )

    st.divider()

    st.subheader("ROC-AUC Comparison")

    fig, ax = plt.subplots()

    ax.bar(
        results_df["Model"],
        results_df["ROC-AUC"]
    )

    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Model")
    ax.set_title("ROC-AUC Comparison")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    st.pyplot(fig)

    st.divider()

    best_model = results_df.loc[
        results_df["ROC-AUC"].idxmax()
    ]

    st.success(
        f"Best model based on ROC-AUC: "
        f"{best_model['Model']} "
        f"({best_model['ROC-AUC']:.4f})"
    )

    st.info(
        """
        Recall is particularly important in churn prediction because
        missing a customer who is likely to churn can result in a lost
        retention opportunity.
        """
    )


# ============================================================
# PAGE 5 — INDIVIDUAL PREDICTION
# ============================================================

elif page == "Individual Prediction":

    st.header("🔮 Individual Customer Churn Prediction")

    st.write(
        "Enter customer information to estimate churn probability."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=800.0
            )

        submitted = st.form_submit_button(
            "Predict Churn"
        )

    if submitted:

        # Build input record
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        # Feature engineering
        input_data["TenureGroup"] = pd.cut(
            input_data["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=[
                "0-12",
                "13-24",
                "25-48",
                "49-72"
            ]
        )

        input_data["MonthlyChargesGroup"] = pd.cut(
            input_data["MonthlyCharges"],
            bins=[-1, 30, 60, 90, float("inf")],
            labels=[
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )

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

            input_data[col + "_binary"] = (
                input_data[col]
                .astype(str)
                .str.lower()
                .isin(
                    ["yes", "yes internet service"]
                )
                .astype(int)
            )

        binary_columns = [
            col + "_binary"
            for col in service_columns
        ]

        input_data["TotalServicesUsed"] = (
            input_data[binary_columns].sum(axis=1)
        )

        try:

            probability = model.predict_proba(
                input_data
            )[0][1]

            if probability < 0.30:
                risk = "Low Risk"
                recommendation = (
                    "Continue normal engagement and monitor customer activity."
                )

            elif probability < 0.60:
                risk = "Medium Risk"
                recommendation = (
                    "Consider targeted offers, service support and engagement."
                )

            else:
                risk = "High Risk"
                recommendation = (
                    "Prioritize this customer for proactive retention outreach."
                )

            st.divider()

            col1, col2 = st.columns(2)

            col1.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Risk Category",
                risk
            )

            st.subheader("Retention Recommendation")

            st.write(recommendation)

        except Exception as e:
            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# PAGE 6 — BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "Business Recommendations":

    st.header("💡 Business Recommendations")

    st.subheader("Customer Retention Strategy")

    recommendations = [
        (
            "1. Identify high-risk customers early",
            "Use predicted churn probabilities to prioritize customers "
            "for proactive retention campaigns."
        ),
        (
            "2. Focus on contract-related churn patterns",
            "Analyze customers with different contract types and design "
            "appropriate retention offers based on observed churn rates."
        ),
        (
            "3. Monitor customers with short tenure",
            "Newer customers can be monitored closely during the early "
            "stages of their relationship with the company."
        ),
        (
            "4. Investigate high monthly charges",
            "Customers with higher charges should be analyzed for "
            "value perception and suitable pricing or service offers."
        ),
        (
            "5. Prioritize high-probability churn customers",
            "Customers with high predicted churn probability can be "
            "given priority by the retention team."
        ),
        (
            "6. Use model predictions as decision support",
            "Predictions should support business decisions rather than "
            "replace human judgment."
        )
    ]

    for title, description in recommendations:

        st.markdown(f"### {title}")
        st.write(description)

    st.divider()

    st.subheader("Important Business Note")

    st.info(
        """
        Customer churn prediction identifies patterns associated with
        customers who may leave. The predictions should be combined with
        customer-service information and business context before taking
        retention actions.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning & Data Analytics Portfolio Project"
)