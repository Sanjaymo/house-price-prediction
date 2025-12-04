import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# -----------------------------
# Utilities
# -----------------------------

def safe_float_input(prompt, default=None):
    """Get safe float input from the user with error handling."""
    while True:
        value = input(f"{prompt} (default={default}): ").strip()
        
        if value == "" and default is not None:
            return float(default)
        try:
            return float(value)
        except ValueError:
            print("❌ Invalid input! Enter a number.")


# -----------------------------
# Train & Evaluate Model
# -----------------------------

def train_and_evaluate(test_size=0.2, random_state=42):
    print("\n📌 Loading California Housing Dataset...")
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    Y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("\n📊 TEST RESULTS")
    print("---------------------------")
    print("Mean Squared Error:", mean_squared_error(y_test, y_test_pred))
    print("R² Score:", r2_score(y_test, y_test_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_test_pred))

    print("\n📊 TRAIN RESULTS")
    print("---------------------------")
    print("Mean Squared Error:", mean_squared_error(y_train, y_train_pred))
    print("R² Score:", r2_score(y_train, y_train_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_train, y_train_pred))

    return model, X.columns


# -----------------------------
# Predict Function (User Input)
# -----------------------------

def predict_from_input(model, X_columns):
    print("\n🧮 Enter values for prediction")
    print("Press ENTER to use default values.\n")

    # Default example values
    defaults = {
        "MedInc": 8,
        "HouseAge": 20,
        "AveRooms": 6,
        "AveBedrms": 1.2,
        "Population": 1500,
        "AveOccup": 3,
        "Latitude": 34,
        "Longitude": -118
    }

    user_values = []
    for col in X_columns:
        default = defaults.get(col, 0)
        val = safe_float_input(f"Enter value for '{col}'", default)
        user_values.append(val)

    new_data = pd.DataFrame([user_values], columns=X_columns)

    price = model.predict(new_data)[0]
    print("\n🏡 Predicted Median House Value:", price, " (in $100,000s)")



# -----------------------------
# Actual vs Predicted Graph
# -----------------------------

def show_graph(model):
    print("\n📌 Generating visualization...")

    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    Y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    plt.scatter(y_test, y_test_pred, color='blue', label='Test Data', alpha=0.6)
    plt.scatter(y_train, y_train_pred, color='green', label='Train Data', alpha=0.3)

    all_actual = np.concatenate([y_test.values, y_train.values])
    min_val, max_val = all_actual.min(), all_actual.max()
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Perfect Prediction')

    plt.xlabel("Actual Median House Value ($100,000s)")
    plt.ylabel("Predicted Median House Value ($100,000s)")
    plt.title("California Housing — Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Menu System
# -----------------------------

def menu():
    model, X_columns = train_and_evaluate()

    while True:
        print("\n============================")
        print("        MAIN MENU")
        print("============================")
        print("1. Predict House Value")
        print("2. Show Actual vs Predicted Graph")
        print("3. Retrain Model")
        print("4. Exit")
        print("============================")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            predict_from_input(model, X_columns)
        elif choice == "2":
            show_graph(model)
        elif choice == "3":
            model, X_columns = train_and_evaluate()
        elif choice == "4":
            print("👋 Exiting program...")
            sys.exit()
        else:
            print("❌ Invalid option. Try again.")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    menu()
