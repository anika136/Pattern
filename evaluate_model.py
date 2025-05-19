import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ✅ Load Processed Data
print("📂 Loading preprocessed data for evaluation...")
data_dict = pickle.load(open('data_cnn.pickle', 'rb'))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# ✅ Convert Labels to Numeric (A-Z → 0-25)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# ✅ Load Trained Model
print("📂 Loading trained model...")
model = tf.keras.models.load_model("sign_language_cnn.h5")

# ✅ Make Predictions
print("🔎 Evaluating model...")
predictions = model.predict(data)
predicted_labels = np.argmax(predictions, axis=1)

# ✅ Compute Metrics
accuracy = accuracy_score(labels, predicted_labels)
precision = precision_score(labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(labels, predicted_labels, average='macro', zero_division=0)
f1 = f1_score(labels, predicted_labels, average='macro', zero_division=0)

# ✅ Print Final Results
print("\n✅ FINAL MODEL RESULTS ✅")
print(f"Accuracy   : {accuracy:.5f}")
print(f"Precision  : {precision:.5f}")
print(f"Recall     : {recall:.5f}")
print(f"F1-Score   : {f1:.5f}")

# ✅ Generate Detailed Classification Report
print("\n📊 Classification Report:")
print(classification_report(labels, predicted_labels, target_names=label_encoder.classes_))
