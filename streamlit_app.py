import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡Housing Price Predictor")
st.caption("Multiple Linear Regression model using the Housing dataset (scikit-learn).")


# -----------------------------
# Cached helpers
# -----------------------------
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    return X, y


@st.cache_resource
def train_model(test_size, random_state):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "Train MSE": mean_squared_error(y_train, y_train_pred),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Train R²": r2_score(y_train, y_train_pred),
        "Test MSE": mean_squared_error(y_test, y_test_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Test R²": r2_score(y_test, y_test_pred),
    }

    return model, X, y, metrics


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Test Size (fraction of dataset)",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Random State (for reproducibility)",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.info("Adjust the parameters above to retrain the model automatically.")


# Train model with chosen settings
model, X, y, metrics = train_model(test_size, random_state)


# -----------------------------
# Tabs for UI sections
# -----------------------------
tab_predict, tab_data, tab_metrics = st.tabs(
    ["🔮 Predict House Value", "📊 Dataset Overview", "📈 Model Performance"]
)


# -----------------------------
# 🔮 Prediction Tab
# -----------------------------
with tab_predict:
    st.subheader("Provide Input Features")

    col1, col2 = st.columns(2)

    inputs = {}

    # Split features into two columns for nicer layout
    features = list(X.columns)
    half = len(features) // 2
    left_features = features[:half]
    right_features = features[half:]

    with col1:
        for col in left_features:
            col_min = float(X[col].min())
            col_max = float(X[col].max())
            default = float(X[col].median())
            step = (col_max - col_min) / 100 if col_max > col_min else 0.1

            inputs[col] = st.slider(
                col,
                min_value=col_min,
                max_value=col_max,
                value=default,
                step=step,
            )

    with col2:
        for col in right_features:
            col_min = float(X[col].min())
            col_max = float(X[col].max())
            default = float(X[col].median())
            step = (col_max - col_min) / 100 if col_max > col_min else 0.1

            inputs[col] = st.slider(
                col,
                min_value=col_min,
                max_value=col_max,
                value=default,
                step=step,
            )

    input_df = pd.DataFrame([inputs])

    st.markdown("### Input Summary")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Predict House Price"):
        pred = model.predict(input_df)[0]
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric(
                label="Predicted Median House Value (in $100,000s)",
                value=f"{pred:.3f}",
            )
        with col_b:
            st.metric(
                label="Approximate Value in USD",
                value=f"${pred * 100000:,.2f}",
            )

        st.success("Prediction completed successfully!")


# -----------------------------
# 📊 Dataset Tab
# -----------------------------
with tab_data:
    st.subheader("Dataset Overview")

    st.markdown("**Housing Dataset – First 10 Rows**")
    st.dataframe(X.head(10), use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Feature Summary (X)**")
        st.dataframe(X.describe().T, use_container_width=True)

    with col4:
        st.markdown("**Target (Median House Value) Summary**")
        st.dataframe(y.describe().to_frame("MedHouseVal"), use_container_width=True)


# -----------------------------
# 📈 Metrics Tab
# -----------------------------
with tab_metrics:
    st.subheader("Model Performance")

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    st.dataframe(metrics_df, use_container_width=True)

    # Split metrics nicely
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Train Metrics**")
        st.write(f"- MSE: `{metrics['Train MSE']:.4f}`")
        st.write(f"- MAE: `{metrics['Train MAE']:.4f}`")
        st.write(f"- R²: `{metrics['Train R²']:.4f}`")

    with col6:
        st.markdown("**Test Metrics**")
        st.write(f"- MSE: `{metrics['Test MSE']:.4f}`")
        st.write(f"- MAE: `{metrics['Test MAE']:.4f}`")
        st.write(f"- R²: `{metrics['Test R²']:.4f}`")

    st.markdown("---")
    st.info(
        "Tip: Try changing the *Test Size* and *Random State* in the sidebar to see "
        "how the model performance changes."
    )

