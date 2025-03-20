import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(train_data):
    plt.figure(figsize=(12, 8))
    numeric_data = train_data.select_dtypes(include=[float, int])
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Train Data')
    plt.show()

# Load the dataset
data = pd.read_parquet(r'C:\Users\rewak\git\ghc\private\pythonProject\hackaton\dataset_processed.parquet')

# Print class distribution before limiting
print("Class distribution before limiting:")
print(data['radio'].value_counts())

# Limit each class of the 'radio' column to a maximum of 1,000 samples
data = data.groupby('radio').apply(lambda x: x.sample(n=10000, random_state=42) if len(x) > 10000 else x).reset_index(drop=True)

# Print class distribution after limiting
print("Class distribution after limiting:")
print(data['radio'].value_counts())

# Print columns of the dataset
print("Columns of the dataset:")
print(data.columns)

# Drop unnecessary columns
data = data.drop(columns=['changeable', 'averageSignal', 'updated_ts_n', 'lat'])

# plot_correlation_matrix(data)

# Define features (X) and target (y)
X = data.drop(columns=['radio'])  # Drop the target column
y = data['radio']  # Target column

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print training and testing set sizes
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Initialize the SVM classifier with grid search for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, scoring='recall_macro')
grid.fit(X_train, y_train)

# Save the trained model to an .h5 file
model_path = 'svm_model.h5'
joblib.dump(grid.best_estimator_, model_path)
print(f"Model saved to {os.path.abspath(model_path)}")

# Make predictions
y_pred = grid.predict(X_test)

# Evaluate the model
recall = recall_score(y_test, y_pred, average="macro")
print(f"Model Macro Average Recall: {recall:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Display statistics
print("Statistics:")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")