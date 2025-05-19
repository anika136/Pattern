import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ✅ Load Processed Data
print("📂 Loading preprocessed data for evaluation...")
data_dict = pickle.load(open('data_cnn.pickle', 'rb'))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# ✅ Convert Labels to Numeric
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# ✅ Load Trained Model
print("📂 Loading trained model...")
model = tf.keras.models.load_model("sign_language_cnn.h5")

# ✅ Make Predictions
print("🔎 Evaluating model...")
predictions = model.predict(data)
predicted_labels = np.argmax(predictions, axis=1)

# ✅ Compute Accuracy
accuracy = accuracy_score(labels, predicted_labels)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Generate Classification Report
print("\n📊 Classification Report:")
print(classification_report(labels, predicted_labels, target_names=label_encoder.classes_))
