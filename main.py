import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

# Load data
train_data = pd.read_parquet('dataset.parquet')

# Preprocessing
train_data['created'] = pd.to_datetime(train_data['created'])
train_data['updated'] = pd.to_datetime(train_data['updated'])
train_data['time_diff'] = (train_data['updated'] - train_data['created']).dt.total_seconds()

# Parse lonlat into latitude and longitude
train_data[['lon', 'lat']] = train_data['lonlat'].str.extract(r'([-\d.]+)[°\'"]\w\s([-\d.]+)[°\'"]\w')

# Drop unnecessary columns
train_data = train_data.drop(columns=['created', 'updated', 'lonlat'])

# Encode target variable
label_encoder = LabelEncoder()
train_data['radio'] = label_encoder.fit_transform(train_data['radio'])

# Split data
X = train_data.drop(columns=['radio'])
y = train_data['radio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Macro Average Recall:", recall_score(y_test, y_pred, average="macro"))