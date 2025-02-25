import pandas as pd
import numpy as np # Handles mathematical operations and arrays efficiently. for TensorFlow.
import matplotlib.pyplot as plt #Plots graphs for understanding data distribution and model performance.
import seaborn as sns # a data visualization  stat
from keras.models import Model # Using Model API instead of Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder #encode male and female

# Step 1: Load the dataset and check on its structure
# Note: The dataset has a semicolon separator
file_path = "cardio_train.csv"
df = pd.read_csv(file_path, sep=';')
print(df)

# Checking dataset structure

# Convert age from days to years
df['age'] = df['age'] // 365
print(df[['age']].head())

# Checking dataset shape (rows, columns)
print(df.shape)  

# Checking dataset structure (non-null count, data types, memory usage)
df.info()  

# Checking summary statistics and rounding the values to 1 decimal place (mean, standard deviation, min, max, etc.) (transpose for better readability)
print(df.describe().transpose().round(1))


# Step 2: Data Preprocessing

# Drop missing values (if any, here there is none)
missing_rows = df[df.isnull().any(axis=1)]
print(missing_rows)
df = df.dropna() #Removes all rows that have at least one missing value (NaN).
print(df.shape)  # Check the shape after dropping missing values

# Check for duplicates and remove them
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
df = df.drop_duplicates()
print(df.shape)  # Check the shape after dropping duplicate values

# Dropping the 'id' column as it is not useful for the model training process
df = df.drop(columns=['id'])
print(df.head()) # Display the first few rows to confirm it's dropped

# Convert 'cholesterol' and 'gluc' to categorical with ordered levels
cholesterol_order = pd.CategoricalDtype(categories=[1, 2, 3], ordered=True)
gluc_order = pd.CategoricalDtype(categories=[1, 2, 3], ordered=True)

df['cholesterol'] = df['cholesterol'].astype(cholesterol_order)
df['gluc'] = df['gluc'].astype(gluc_order)

# Checking dataset structure Output after conversion:
df.info()  

#Exploration of data 

#univariate analysis: Visualize the distribution of the target variable 'cardio'
df['cardio'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of Target Variable (Cardio)")
plt.xlabel("Cardio Status (0 = No Cardiovascular Disease, 1 = Cardiovascular Disease)")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.show()

#bivariate analysis: show gnder
"""
sns.catplot(data=df, x='gender', y='cardio', kind='bar')
plt.title('Cardiovascular Disease by Gender')
plt.xlabel('Gender')
plt.ylabel('Proportion with Cardiovascular Disease')
plt.show()
"""
#Mapping gender values to labels for better readability
# Encode the 'gender' column before scaling
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])  # Converts 'Male' to 0 and 'Female' to 1
  # <-- Modified this part

# Replot with updated labels
sns.catplot(data=df, x='gender', y='cardio', kind='bar')
plt.title('Cardiovascular Disease by Gender')
plt.xlabel('Gender')
plt.ylabel('Proportion with Cardiovascular Disease')
plt.show()



# defining x and y
X = df.drop(columns=['cardio']).values  # Drop target column (features)
y = df['cardio'].values  # Target labels

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM input (samples, timesteps, features)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Build LSTM Model with Encoder-Decoder Architecture
""" Instead of using a Sequential model, we now use an Encoder-Decoder architecture. """

# Input Layer
inputs = Input(shape=(1, X_train.shape[2]))

# 📌 Encoder (Feature Extraction)
encoder = LSTM(64, return_sequences=True)(inputs)  
encoder = Dropout(0.3)(encoder)
encoder = LSTM(32, return_sequences=False)(encoder)  # Extracts final features

# 📌 Decoder (Processes Extracted Features)
decoder = Dense(16, activation='relu')(encoder)  # More feature extraction
#Takes the compressed feature vector (encoder) and further processes it.
decoder = Dropout(0.3)(decoder)
outputs = Dense(1, activation='sigmoid')(decoder)  # Binary classification (0 or 1)
# Final Output of decoder:A single probability value representing the likelihood of cardiovascular disease.
# Create the model
model = Model(inputs, outputs)

""" 
- Encoder extracts features using LSTM layers.
- Decoder refines extracted features before final classification.
"""

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#DISPLAY SUMMARY (TABLE)
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Evaluate the model
# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot accuracy and loss 
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

"""
diff btwn LSTM and Encoder, decoder LSTM

LSTM:
-No explicit feature extraction mechanism.
-It doesn’t separate encoding and decoding steps--> struggle with complex relationships in the data.

DECODER- ENCODER LSTM:

-Feature extraction is handled by the encoder before classification.
-More structured representation of data.
-Better for learning complex patterns compared to a basic stacked LSTM.


SYNTEX:

#LSTM
 LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])),#defines time steps = 1 and features = X_train.shape[2].
 ## extracts features from input data and keeps passing the sequence to the next LSTM layer
       Dropout(0.3),
    LSTM(32), #does not return sequences by default

# Encoder
encoder = LSTM(64, return_sequences=True)(inputs) #Takes the inputs and outputs a sequence (used by the next LSTM).
encoder = Dropout(0.3)(encoder)
encoder = LSTM(32, return_sequences=False)(encoder)  # Does not return for seq ;; false


** model = Model(inputs, outputs) Instead of Sequential([...])
it Connects inputs → encoder → decoder → outputs to form a complete model.
"""