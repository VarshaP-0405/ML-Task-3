#Import required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load Data: Import CSV using pandas.
df = pd.read_csv("car_data.csv")   
print(df.head())

# Explore: Check missing values, descriptive statistics (df.describe()), and scatter plots to observe non-linear relationships.
print(df.info())
print(df.isnull().sum())
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.fillna(df.mean(numeric_only=True), inplace=True)
print(df.describe())
sns.scatterplot(x=df['weight'], y=df['mpg'])
plt.title("Weight vs MPG")
plt.show()
sns.scatterplot(x=df['horsepower'], y=df['mpg'])
plt.title("Horsepower vs MPG")
plt.show()

# Prepare: Select features (Displacement, Horsepower, Weight, Acceleration) and target (MPG), split train/test (80/20).
X = df[['displacement', 'horsepower', 'weight', 'acceleration']]
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
                                                   )
# Transform Features: Use PolynomialFeatures(degree=2) from sklearn. preprocessing to capture non-linear patterns.
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Build Model: Use LinearRegression() on transformed polynomial features and fit on training data.
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

# Evaluate: Predict MPG on test set, calculate Mean Squared Error (MSE) and R² score.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Interpret: Analyze the impact of features, and plot predicted vs actual MPG values to visualize model performance.
print("Number of features after polynomial transform:", len(model.coef_))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()
