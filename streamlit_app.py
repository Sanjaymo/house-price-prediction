import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Housing Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡 Housing Price Predictor")
st.caption(
    "Predicting the **Median House Value (USD)** using a "
    "Multiple Linear Regression model (California Housing Dataset)."
)


# -----------------------------
# Load Dataset (Safe Mode)
# -----------------------------
@st.cache_data
def load_data():
    try:
        housing = fetch_california_housing(as_frame=True)

        X = housing.data
        y = housing.target

        return X, y

    except Exception:
        st.error(
            "❌ Unable to download the California Housing dataset.\n\n"
            "This app uses `fetch_california_housing()` which requires "
            "internet access. Streamlit Cloud blocks runtime downloads.\n\n"
            "✅ To fix permanently: download the dataset locally and include "
            "it as a CSV file in the project."
        )
        st.stop()


# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model(test_size, random_state):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    metrics = {
        "Train MSE": mean_squared_error(y_train, model.predict(X_train)),
        "Train MAE": mean_absolute_error(y_train, model.predict(X_train)),
        "Train R²": r2_score(y_train, model.predict(X_train)),
        "Test MSE": mean_squared_error(y_test, y_test_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Test R²": r2_score(y_test, y_test_pred),
    }

    return model, X, y, metrics


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Test Size (fraction of dataset)",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random State (for reproducibility)",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)


# -----------------------------
# Train Model
# -----------------------------
model, X, y, metrics = train_model(test_size, random_state)


# -----------------------------
# Tabs
# -----------------------------
tab_predict, tab_data, tab_metrics = st.tabs(
    ["🔮 Predict House Value", "📊 Dataset Overview", "📈 Model Performance"]
)


# -----------------------------
# Prediction Tab
# -----------------------------
with tab_predict:
    st.subheader("Provide Input Features")

    col1, col2 = st.columns(2)
    inputs = {}

    features = list(X.columns)
    half = len(features) // 2
    left_features = features[:half]
    right_features = features[half:]

    with col1:
        for col in left_features:
            min_val = int(X[col].min())
            max_val = int(X[col].max())
            default = int(X[col].median())

            inputs[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=1
            )

    with col2:
        for col in right_features:
            min_val = int(X[col].min())
            max_val = int(X[col].max())
            default = int(X[col].median())

            inputs[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=1
            )

    input_df = pd.DataFrame([inputs])

    st.markdown("### 🧾 Input Summary")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Predict House Price"):
        pred = model.predict(input_df)[0]
        pred = max(pred, 0)

        st.success("Prediction completed successfully!")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric(
                "Predicted Median House Value (USD)",
                f"${pred * 100000:,.2f}"
            )

        with col_b:
            st.caption(
                "ℹ️ Prediction is based on the California Housing dataset "
                "where values are scaled in units of $100,000."
            )


# -----------------------------
# Dataset Tab
# -----------------------------
with tab_data:
    st.subheader("Dataset Overview")

    st.markdown("**First 10 Rows**")
    st.dataframe(X.head(10), use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Feature Summary**")
        st.dataframe(X.describe().T, use_container_width=True)

    with col4:
        st.markdown("**Target Summary**")
        st.dataframe(y.describe().to_frame("Median House Value"), use_container_width=True)


# -----------------------------
# Metrics Tab
# -----------------------------
with tab_metrics:
    st.subheader("Model Performance")

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    st.dataframe(metrics_df, use_container_width=True)

    st.info(
        "Try adjusting the Test Size and Random State from the sidebar "
        "to observe changes in model performance."
    )
