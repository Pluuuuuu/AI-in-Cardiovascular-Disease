

import pandas as pd
import numpy as np # Handles mathematical operations and arrays efficiently. for TensorFlow.
import matplotlib.pyplot as plt #Plots graphs for understanding data distribution and model performance.
import seaborn as sns # a data visualization  stat
from keras.models import Sequential #Defines and builds deep learning models in a sequential way.
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Drop missing values (if any, here there is none)  --> removes rows with missing values. *Note: This is not the best practice for handling missing data, we can use imputation techniques like mean, median, or mode.
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

#categorical variables can and should be included in modeling! The key thing is that most machine learning models require numerical input. So, if you have a categorical variable (like gender, cholesterol with levels Low, Normal, High), you need to convert it into a numerical format before including it in the model.
# we need to inform model that  variables (like choelstero, gluc) are not continuous numerical variables, but ordinal variables: You can define the ordinal relationship using pandas' Categorical type
# The model will treat these variables with their order in mind, rather than treating them as continuous numeric features.

# Step 1: Check unique values for cholesterol and gluc
print("Unique values for cholesterol:", df['cholesterol'].unique())
print("Unique values for gluc:", df['gluc'].unique())

# Step 2: Convert 'cholesterol' and 'gluc' to categorical with ordered levels
cholesterol_order = pd.CategoricalDtype(categories=[1, 2, 3], ordered=True)
gluc_order = pd.CategoricalDtype(categories=[1, 2, 3], ordered=True)

# Step 3: Assign these types to the relevant columns
df['cholesterol'] = df['cholesterol'].astype(cholesterol_order)
df['gluc'] = df['gluc'].astype(gluc_order)

# Verify changes
print("Cholesterol dtype:", df['cholesterol'].dtype)
print("Cholesterol categories:", df['cholesterol'].cat.categories)

print("Gluc dtype:", df['gluc'].dtype)
print("Gluc categories:", df['gluc'].cat.categories)

# Check the first few rows to see the changes
print(df.head())

# Checking dataset structure Output after conversion: When you call df.info(), you can see that both cholesterol and gluc have the category dtype with the correct ordered categories, so the model knows how to interpret them.
df.info()  

#Exploration of data 

#univariate analysis: Visualize the distribution of the target variable 'cardio'
df['cardio'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of Target Variable (Cardio)")
plt.xlabel("Cardio Status (0 = No Cardiovascular Disease, 1 = Cardiovascular Disease)")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.show()

#bivariate analysis:
sns.catplot(data=df, x='gender', y='cardio', kind='bar')
# Add a title and labels to the axes
plt.title('Cardiovascular Disease by Gender')
plt.xlabel('Gender')
plt.ylabel('Proportion with Cardiovascular Disease')
# Display the plot
plt.show()

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




