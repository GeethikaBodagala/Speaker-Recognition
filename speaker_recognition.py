import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Function to load audio files and extract MFCC features
def extract_mfcc_features(file_path, max_pad_len=400):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load data
def load_data(data_path, max_files_per_speaker=100):
    X = []
    y = []
    speakers = os.listdir(data_path)
    for speaker in speakers:
        speaker_dir = os.path.join(data_path, speaker)
        if os.path.isdir(speaker_dir):
            for subdir, _, files in os.walk(speaker_dir):
                for file in files[:max_files_per_speaker]:
                    if file.endswith('.wav'):
                        file_path = os.path.join(subdir, file)
                        mfcc = extract_mfcc_features(file_path)
                        if mfcc is not None:
                            X.append(mfcc)
                            y.append(speaker)
    return np.array(X), np.array(y)

# Set the path to the extracted dataset
data_path = '/path/to/dataset'  # Modify this to your dataset path

# Load data and extract MFCC features
X, y = load_data(data_path)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, 5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Function to predict speaker
def predict_speaker(file_path):
    mfcc = extract_mfcc_features(file_path)
    if mfcc is not None:
        mfcc = np.expand_dims(mfcc, axis=0)
        prediction = model.predict(mfcc)
        speaker_id = le.inverse_transform([np.argmax(prediction)])
        return speaker_id[0]
    else:
        return None

# Example usage: predicting the speaker of a test audio file
test_file_path = os.path.join(data_path, 's1', 'test_speaker(1).wav')
identified_speaker = predict_speaker(test_file_path)
print(f'Identified Speaker: {identified_speaker}')
