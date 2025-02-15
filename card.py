import pandas as pd
import numpy as np # Handles mathematical operations and arrays efficiently. for TensorFlow.
import matplotlib.pyplot as plt #Plots graphs for understanding data distribution and model performance.
import seaborn as sns # a data visualization  stat
from keras.models import Sequential #Defines and builds deep learning models in a sequential way.
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
# Note: The dataset has a semicolon separator
file_path = "cardio_train.csv"
df = pd.read_csv(file_path, sep=';')

# Step 2: Data Preprocessing
# Drop missing values
df = df.dropna() #Removes all rows that have at least one missing value (NaN).

# Convert age from days to years
df['age'] = df['age'] // 365

# Separate input (X) and output (y)
X = df.drop(columns=['cardio'])  # All features except the target Use 'cardio' as the target
y = df['cardio']# Target variable (0 = No CVD, 1 = CVD)

# Normalize data
#using StandardScaler from sklearn.preprocessing.
#all features have the same scale, improving model performance.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #all features will have similar ranges → better for training the LSTM model!

# Reshape for LSTM (samples, time steps, features)
""" LSTM requires 3D input with the format:
(samples, time steps, features)
-samples → Number of rows (patients).
-time steps → Number of time points per sample (here, 1).
-features → Number of features (columns in X).
LSTMs work with sequential time-series data  --> we only have one timestamp per patient (not a sequence). --> we reshape data with 1 time step. (1 time stamp, several features)
""" 
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split dataset into training and testing
""" Splits the data into 80% training and 20% testing.
-stratify=y ensures that the distribution of cardio (0 and 1) is the same in both train and test sets.
-random_state=42 → Ensures reproducibility (same split every time)""" 
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Build LSTM Model
model = Sequential([
    #Sequential() creates a stack of layers, meaning each layer feeds into the next one|common way to define deep learning models in Keras.
    #Extracts patterns from data (returns sequences)
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])),
    # extracts features from input data and keeps passing the sequence to the next LSTM layer.
    Dropout(0.3), #prevent overrfitting by randomly disabling 30% of neurons during training. -->forces the model to learn robust patterns.
    #2nd LSTM layer
    #Extracts deeper patterns (returns only final output)
    LSTM(32), #has 32 memory cells. does not return sequences by default
    Dropout(0.3), #Makes sure the model generalizes well.
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification 
        # this layer has 1 neuron because we are doing binary classification --> Converts the processed LSTM output into a probability for classification.
    ])
""" -LSTM(64) → This layer has 64 memory cells (neurons).
        -input_shape=(1, X_train.shape[2])
            1 → Time step (since we only have one data point per patient).
            X_train.shape[2] → Number of features (11 features in our dataset).
        -return_sequences=True
            Ensures that this LSTM passes output sequences to the next LSTM layer.
            Without this, only the last time step would be passed."""
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
"""compile() function sets up how the model will learn by defining:
        Optimizer (adam) → How the model updates weights | to minimize the loss function
        Loss Function (binary_crossentropy) → How the model measures errors
        Metrics (accuracy) → How we evaluate performance  """

#DISPLAY SUMMARY (TABLE)
model.summary()
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
# model learns patterns from training data and adjusts its weights.
"""fit() function starts training the model.
    -X_train, y_train → Training data and labels.
    -epochs=50 The model sees the entire dataset 50 times.
         More epochs → better learning but risk of overfitting.
    -batch_size=32
        Model updates weights after every 32 samples.
        Small batch sizes (e.g., 16) → better learning, but slower.
        Large batch sizes (e.g., 128) → faster, but less accurate.
    -validation_data=(X_test, y_test)
        Evaluates performance on unseen test data after every epoch.
        Helps check if the model is overfitting"""
# Step 4: Evaluate the model
# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot accuracy and loss 
##acc-left shows how well the model's predictions match the actual values
#loss-right shows how much error (loss) the model is making during training and validation.
    #Train Loss: Loss on the training set.
    #Validation Loss: Loss on the validation set.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()



#commit
