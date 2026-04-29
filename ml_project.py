import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create simple dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Scores": [10,20,30,40,50,60,70,80,90,100]
}

df = pd.DataFrame(data)

print("\n--- DATASET ---\n")
print(df)

# Split data
X = df[["Hours_Studied"]]
y = df["Scores"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
hours = float(input("\nEnter study hours: "))
prediction = model.predict([[hours]])

print(f"\nPredicted Score: {prediction[0]:.2f}")