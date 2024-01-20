import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load data from a text file
url = os.path.join(os.getcwd(), "Dataset\\data.txt")
data = np.loadtxt(url, delimiter=',', skiprows=1, dtype=float, encoding='utf-8')

# Separate features (X) and labels (y)
X_train = data[:, :2]
y_train = data[:, 2]

pos_label = "y=1"
neg_label = "y=0"

# Plot examples without decision boundary
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))

pos = y_train == 1
neg = y_train == 0

ax1.scatter(X_train[pos, 0], X_train[pos, 1], marker='o', s=80, c='blue', label=pos_label)
ax1.scatter(X_train[neg, 0], X_train[neg, 1], marker='x', s=80, c='red', label=neg_label)
ax1.legend(loc='best')

ax1.figure.canvas.toolbar_visible = False
ax1.figure.canvas.header_visible = False
ax1.figure.canvas.footer_visible = False

ax1.set_xlabel('x0')
ax1.set_ylabel('x1')
plt.title("Logistic Regression - Data Plot")

# Save the first plot to a PNG file
fig1.savefig('data_plot.png')
plt.close(fig1)

# Fit logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Get coefficients and intercept
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]

# Visualization of decision boundary
x0 = np.linspace(min(X_train[:, 0]) - 5, max(X_train[:, 0]) + 5, 100)
x1 = -(coefficients[0] * x0 + intercept) / coefficients[1]

# Plot decision boundary separately
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))

ax2.plot(x0, x1, c='green', label="Decision Boundary")
ax2.scatter(X_train[pos, 0], X_train[pos, 1], marker='o', s=80, c='blue', label=pos_label)
ax2.scatter(X_train[neg, 0], X_train[neg, 1], marker='x', s=80, c='red', label=neg_label)
ax2.legend(loc='best')

ax2.figure.canvas.toolbar_visible = False
ax2.figure.canvas.header_visible = False
ax2.figure.canvas.footer_visible = False

ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
plt.title("Logistic Regression - Decision Boundary")

# Save the second plot to a PNG file
fig2.savefig('decision_boundary.png')
plt.close(fig2)  # Close the second plot


