import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the dataset
# Replace 'your_dataset.csv' with the actual dataset file path
data = pd.read_parquet('dataset_processed.parquet')

# Define features (X) and target (y)
X = data.drop(columns=['Radio'])  # Drop the target column
y = data['Radio']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm = SVC(kernel='linear', random_state=42)  # You can adjust the kernel (e.g., 'rbf', 'poly') as needed

# Train the model
svm.fit(X_train, y_train)

# Save the trained model to an .h5 file
model_path = 'svm_model.h5'
joblib.dump(svm, model_path)
print(f"Model saved to {os.path.abspath(model_path)}")

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Display statistics
print("Statistics:")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")