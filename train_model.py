import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ✅ Load Processed Data
print("📂 Loading preprocessed data...")
data_dict = pickle.load(open('data_cnn.pickle', 'rb'))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# ✅ Convert Labels to Numeric (A-Z → 0-25)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# ✅ Split Dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# ✅ Define CNN Model
print("🛠️ Building the CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # 26 output classes (A-Z)
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Train Model
print("🚀 Training the model...")
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# ✅ Save Model
model.save("sign_language_cnn.h5")
print("🎉 Model training complete! Saved as 'sign_language_cnn.h5'.")