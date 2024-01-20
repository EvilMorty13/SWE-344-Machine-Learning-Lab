import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load your CSV dataset
url = os.path.join(os.getcwd(), "Dataset\\4K_House_Rent_Dataset.csv")
print(url)
df = pd.read_csv(url)

print(df.head())

# Select only the relevant features
selected_feature = "Size"
df_selected = df[[selected_feature, "Rent"]]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_selected[[selected_feature]])

# Split the data
X = np.column_stack((np.ones(len(X_scaled)), X_scaled))  # Add a column of ones for the bias term
y = df_selected["Rent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Descent for Simple Linear Regression
def hypothesis(theta, X):
    return np.dot(X, theta)

def cost_function(theta, X, y):
    m = len(y)
    h = hypothesis(theta, X)
    return (1 / (2 * m)) * np.sum((h - y) ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        h = hypothesis(theta, X)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        cost_history.append(cost_function(theta, X, y))

    return theta, cost_history

# Initialize theta and set hyperparameters
theta_init = np.zeros(X_train.shape[1])
alpha = 0.01
iterations = 1000

# Run gradient descent
theta_final, cost_history = gradient_descent(X_train, y_train.values, theta_init, alpha, iterations)

# Make predictions on the test set
X_test_with_bias = np.column_stack((np.ones(len(X_test)), X_test[:, 1:]))  # Add a column of ones for the bias term
y_pred = hypothesis(theta_final, X_test_with_bias)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Create a single plot for both graphs
plt.figure(figsize=(12, 6))

# Plot Rent vs Size with Regression Line
plt.subplot(1, 2, 1)
plt.scatter(df_selected[selected_feature], df_selected["Rent"], color='blue', label='Actual Rent')
plt.plot(df_selected[selected_feature], hypothesis(theta_final, X), color='red', linewidth=2, label='Regression Line')
plt.xlabel(selected_feature)
plt.ylabel('Rent')
plt.title('Rent vs Size with Regression Line')
plt.legend()

# Plot the cost over iterations
plt.subplot(1, 2, 2)
plt.plot(range(1, iterations + 1), cost_history, color='blue')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent: Cost over Iterations')

# Save the combined plot as PNG
plt.savefig("combined_plot.png")

plt.tight_layout()
plt.show()
