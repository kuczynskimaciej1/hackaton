import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import recall_score
import joblib
import os

# Load the dataset
data = pd.read_parquet(r'C:\Users\rewak\git\ghc\private\pythonProject\hackaton\dataset_processed.parquet')

# Limit each class of the 'radio' column to a maximum of 1,000 samples
data = data.groupby('radio').apply(lambda x: x.sample(n=1000, random_state=42) if len(x) > 1000 else x).reset_index(drop=True)

# Define features (X) and target (y)
X = data.drop(columns=['radio'])  # Drop the target column
y = data['radio']  # Target column

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
recall = recall_score(y_test, y_pred, average="macro")
print(f"Model Macro Average Recall: {recall:.2f}")

# Display statistics
print("Statistics:")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")