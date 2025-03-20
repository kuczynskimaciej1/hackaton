import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score
import joblib
import os

# Load the dataset
# Replace 'your_dataset.csv' with the actual dataset file path
data = pd.read_parquet('dataset_processed.parquet')

# Define features (X) and target (y)
X = data.drop(columns=['radio'])  # Drop the target column
y = data['radio']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the GaussianNB classifier
nb = GaussianNB()

# Train the model
nb.fit(X_train, y_train)

# Save the trained model to an .h5 file
model_path = 'naive_bayes_model.h5'
joblib.dump(nb, model_path)
print(f"Model saved to {os.path.abspath(model_path)}")

# Make predictions
y_pred = nb.predict(X_test)

# Evaluate the model
recall_scr = recall_score(y_test, y_pred, average='micro')
print(f"Model Recall score: {recall_scr:.2f}")

# Display statistics
print("Statistics:")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")
