import os
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense

# Load data from a text file
url = os.path.join(os.getcwd(), "Dataset\\nn_data.txt")
data = pd.read_csv(url, header=None)


# Separate features (X) and target variable (y)
X = data.iloc[:, 0:8]
y = data.iloc[:, 8]

# Scale features
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(8, input_shape=(8,), activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train, y_train, epochs=300, batch_size=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
_, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy: %.2f%%' % (accuracy * 100))

# Make predictions on the test set
predictions = (model.predict(X_test) > 0.5).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(cm)

# Display classification report
print('\nClassification Report:')
print(classification_report(y_test, predictions))

# Save confusion matrix and final test accuracy to a text file
with open('model_performance.txt', 'w') as file:
    file.write('Final Test Accuracy: %.2f%%\n' % (accuracy * 100))
    file.write('\nConfusion Matrix:\n')
    file.write(str(cm))

    file.write('\n\nClassification Report:\n')
    file.write(classification_report(y_test, predictions))