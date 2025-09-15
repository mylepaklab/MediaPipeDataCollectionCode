import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import zipfile

# 1. Load data from CSV
csv_path = "combined_clean_gesture_data.csv"  # your updated CSV file name
df = pd.read_csv(csv_path, low_memory=False)

# Assume last column is label, all others are features
X = df.iloc[:, :-1].astype(np.float32).values  # ensure numeric features
y = df.iloc[:, -1].astype(str).values           # labels as strings

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 2. Define a simple model (adjust architecture to your data)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 4. Save Keras model
h5_path = "gesture_model.h5"
model.save(h5_path)
print(f"Model saved as {h5_path}")

# 5. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = "gesture_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model saved as {tflite_path}")

# 6. Save labels to file
labels_path = "labels.txt"
with open(labels_path, "w") as f:
    for label in label_encoder.classes_:
        f.write(label + "\n")
print(f"Labels saved as {labels_path}")

# 7. Create metadata file (dummy example)
metadata_content = f"task: gesture recognition\ninput_shape: {X_train.shape[1]}\nnum_classes: {len(label_encoder.classes_)}\n"
task_path = "model.task"
with open(task_path, "w") as f:
    f.write(metadata_content)
print(f"Metadata saved as {task_path}")

# 8. Zip all files
zip_path = "gesture_model.task.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    zipf.write(tflite_path, arcname="gesture_model.tflite")
    zipf.write(labels_path, arcname="labels.txt")
    zipf.write(task_path, arcname="model.task")

print(f"Packaged files into {zip_path}")

