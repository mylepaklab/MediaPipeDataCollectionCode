import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load cleaned combined CSV and force 'label' column to be string
df = pd.read_csv("combined_clean_gesture_data.csv", dtype={'label': str})

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"].astype(str).str.upper()  # ensure uniform string format

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and encoder
joblib.dump(model, "gesture_model.pkl")
joblib.dump(label_encoder, "gesture_labels.pkl")
print("✅ Model and encoder saved.")

