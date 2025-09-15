import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("combined_clean_gesture_data.csv", low_memory=False)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].astype(str).values  # convert labels to string explicitly

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(model, "gesture_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Saved model and label encoder.")

