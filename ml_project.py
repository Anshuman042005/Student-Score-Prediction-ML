"""
Student Score Prediction using Machine Learning
Author: Anshuman Sikdar
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_data():
    data = {
        "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
        "Scores": [10,20,30,40,50,60,70,80,90,100]
    }
    return pd.DataFrame(data)


def train_model(df):
    X = df[["Hours_Studied"]]
    y = df["Scores"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy (R² Score): {accuracy:.2f}")

    return model


def predict_score(model):
    try:
        hours = float(input("\nEnter study hours: "))
        prediction = model.predict([[hours]])
        print(f"\n📊 Predicted Score: {prediction[0]:.2f}")
    except ValueError:
        print("❌ Please enter a valid number.")


def main():
    print("\n===== STUDENT SCORE PREDICTION =====\n")

    df = load_data()
    print("Dataset:\n", df)

    model = train_model(df)
    predict_score(model)

    print("\n===== DONE =====\n")


if __name__ == "__main__":
    main()
