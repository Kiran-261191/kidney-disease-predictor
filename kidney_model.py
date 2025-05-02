import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and clean dataset
df = pd.read_csv("kidney_disease.csv")
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# Convert types if needed
for col in df.columns:
    if df[col].dtype == object and col != 'classification':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode target variable
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

# Drop rows with NA (simple approach)
df.dropna(inplace=True)

X = df.drop(columns=['classification'])
y = df['classification']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, 'kidney_model.pkl')
