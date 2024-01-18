import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load the dataset
url = os.path.join(os.getcwd(), "Dataset\\bank-full.csv")
df = pd.read_csv(url, sep=';')

# Convert categorical variables into numerical using one-hot encoding
df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'], drop_first=True)

# Convert the target variable 'y' into binary (0 or 1)
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Separate features (X) and target variable (y)
X = df.drop('y', axis=1)
y = df['y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Save the display results to a txt file
results_txt_path = os.path.join(os.getcwd(), "model_evaluation_results.txt")
with open(results_txt_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n")
    f.write("Classification Report:\n")
    f.write(classification_rep)

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the plot to a PNG file
roc_curve_path = os.path.join(os.getcwd(), "roc_curve.png")
plt.savefig(roc_curve_path)
plt.close()

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
