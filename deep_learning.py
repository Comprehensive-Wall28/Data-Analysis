import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# --- Import TensorFlow/Keras for Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping # Optional: for early stopping

# --- Load Data ---
df = pd.read_csv('employee_attrition_dataset.csv')
print("Dataset loaded.")

# --- PREPARATION ---

df_prepared = df.drop('Employee_ID', axis=1)

# --- Label Encoding ---
columns_to_encode = ['Attrition', 'Gender']
label_encoder_attrition = LabelEncoder()
label_encoder_gender = LabelEncoder()

print("--- Label Encoding ---")
# Encode Attrition
original_values = df_prepared['Attrition'].unique()
df_prepared['Attrition'] = label_encoder_attrition.fit_transform(df_prepared['Attrition'])
encoded_values = df_prepared['Attrition'].unique()
mapping_attrition = dict(zip(label_encoder_attrition.classes_, label_encoder_attrition.transform(label_encoder_attrition.classes_)))
print(f"  Attrition mapping: {mapping_attrition}")


# Encode Gender
original_values = df_prepared['Gender'].unique()
df_prepared['Gender'] = label_encoder_gender.fit_transform(df_prepared['Gender'])
encoded_values = df_prepared['Gender'].unique()
mapping_gender = dict(zip(label_encoder_gender.classes_, label_encoder_gender.transform(label_encoder_gender.classes_)))
print(f"  Gender mapping: {mapping_gender}")


# Rename the encoded column to Gender_Male IF Male is encoded as 1
if 'Male' in mapping_gender and mapping_gender['Male'] == 1:
        df_prepared.rename(columns={'Gender': 'Gender_Male'}, inplace=True)
        print("  Renamed encoded 'Gender' column to 'Gender_Male'.")
elif 'Female' in mapping_gender and mapping_gender['Female'] == 1:
        # If Female is 1, we need to flip the bits to get Gender_Male (where Male=1)
        print("  Adjusting 'Gender' encoding to create 'Gender_Male' (Male=1).")
        df_prepared['Gender_Male'] = 1 - df_prepared['Gender']
        df_prepared.drop('Gender', axis=1, inplace=True)
elif 'Gender_Male' not in df_prepared.columns:
     # Handle cases where gender might already be numerical or have different labels
     print("Warning: Could not automatically create 'Gender_Male'. Check Gender encoding.")
     # As a fallback, let's assume the existing encoded 'Gender' might represent Male=1 if only two values exist
     if len(df_prepared['Gender'].unique()) == 2 and 1 in df_prepared['Gender'].unique():
         print("  Assuming existing 'Gender' column represents Male=1. Renaming.")
         df_prepared.rename(columns={'Gender': 'Gender_Male'}, inplace=True)
     else:
         print("  Could not determine Gender_Male mapping. Dropping 'Gender'.")
         if 'Gender' in df_prepared.columns:
             df_prepared.drop('Gender', axis=1, inplace=True)


# --- One-Hot Encoding ---
print("\n--- One-Hot Encoding ---")
columns_to_encode_onehot = ['Department'] # Job_Role, Marital_Status removed
columns_exist_for_onehot = [col for col in columns_to_encode_onehot if col in df_prepared.columns]

if columns_exist_for_onehot:
    df_prepared = pd.get_dummies(df_prepared, columns=columns_exist_for_onehot, drop_first=False, prefix='Department') # Added prefix
    print(f"  One-Hot Encoding applied to: {columns_exist_for_onehot}")
else:
    print("  No specified columns found for One-Hot Encoding.")


# ---Undersampling Implementation ---
print("\n--- Performing Undersampling ---")
random_seed = 42
tf.random.set_seed(random_seed) # Set seed for TensorFlow reproducibility
np.random.seed(random_seed) # Ensure numpy operations are also seeded if needed elsewhere

undersampling_ratio = 1.0 #MODIFY THIS VALUE TO CONTROL THE LEVEL (e.g., 1.0, 1.5, 2.0)

# Separate majority and minority classes (Attrition: No=0, Yes=1)
df_majority = df_prepared[df_prepared.Attrition == 0]
df_minority = df_prepared[df_prepared.Attrition == 1]

minority_size = len(df_minority)
majority_size = len(df_majority)

print(f"Original distribution: Majority={majority_size}, Minority={minority_size}")
print(f"Target undersampling ratio (Majority:Minority): {undersampling_ratio}:1")

if majority_size < minority_size:
     print("Warning: Initial 'majority' class (Attrition=0) is smaller than 'minority' class (Attrition=1). "
           "Undersampling logic might behave unexpectedly or not reduce the majority.")
     # In this unusual case, maybe we don't undersample the 'majority' (0)
     # Or adjust logic based on specific needs. Here, we'll proceed but it won't reduce class 0.
     desired_majority_samples = int(minority_size * undersampling_ratio)
     n_samples_majority = min(desired_majority_samples, majority_size)

else:
     # Calculate the desired number of majority samples based on the minority size and the ratio
     desired_majority_samples = int(minority_size * undersampling_ratio)
     # Ensure we don't try to sample more majority samples than actually exist
     n_samples_majority = min(desired_majority_samples, majority_size)

print(f"Calculated samples to keep from majority class: {n_samples_majority}")

# Undersample the majority class to the calculated size
df_majority_undersampled = df_majority.sample(n=n_samples_majority, random_state=random_seed)

# Combine the minority class with the undersampled majority class
df_undersampled = pd.concat([df_majority_undersampled, df_minority])

# Shuffle the resulting DataFrame
df_undersampled = df_undersampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)

print("Final dataset 'Attrition' distribution after undersampling:\n", df_undersampled.Attrition.value_counts())
final_minority_count = len(df_undersampled[df_undersampled.Attrition == 1])
if final_minority_count > 0:
    final_ratio = len(df_undersampled[df_undersampled.Attrition == 0]) / final_minority_count
    print(f"Final ratio (Majority/Minority): {final_ratio:.2f}:1")
else:
    print("Final ratio calculation skipped: No minority samples after processing.")


# --- END: Undersampling Implementation ---

# --- START: Feature Selection (Select Specific Features) ---
print("\n--- Feature Selection (Selecting Specific Features) ---")

# Define the list of features we want to use for modeling.
# Make sure the one-hot encoded columns are included correctly
selected_features = [
    # Original features kept
    'Age',
    'Years_Since_Last_Promotion',
    'Work_Life_Balance',
    'Performance_Rating',
    'Training_Hours_Last_Year',
    'Average_Hours_Worked_Per_Week',
    'Absenteeism',
    'Job_Involvement'
]

# Add Gender_Male if it exists
if 'Gender_Male' in df_prepared.columns:
    selected_features.append('Gender_Male')
    print("  Including 'Gender_Male' in selected features.")
elif 'Gender' in df_prepared.columns and len(df_prepared['Gender'].unique()) == 2:
     # Fallback if renaming failed but 'Gender' is binary
     print("  Warning: 'Gender_Male' not found, but binary 'Gender' exists. Assuming it represents Male=1 and including.")
     df_prepared.rename(columns={'Gender': 'Gender_Male'}, inplace=True) # Try renaming again just in case
     selected_features.append('Gender_Male')


# Add one-hot encoded Department columns if they exist
department_cols = [col for col in df_prepared.columns if col.startswith('Department_')]
if department_cols:
    selected_features.extend(department_cols)
    print(f"  Including one-hot encoded Department columns: {department_cols}")

# Verify that all selected features actually exist in the dataframe before proceeding
final_selected_features = [col for col in selected_features if col in df_prepared.columns]
missing_features = set(selected_features) - set(final_selected_features)
if missing_features:
    print(f"Warning: The following selected features were not found in the prepared dataframe and will be excluded: {missing_features}")
selected_features = final_selected_features
print(f"Final features used for modeling: {selected_features}")


# --- Define Target (y) and Features (X) from Undersampled Data ---
print("\n--- Defining Target (y) and Features (X) ---")
target_variable = 'Attrition'
print("Use undersampled data for modeling? (y / n)")
ans = input()
if ans.lower() == 'y':
    print("Using undersampled data.")
    current_df = df_undersampled
else:
    print("Using original (prepared) data.")
    current_df = df_prepared

# Ensure all selected features exist in the chosen dataframe (current_df)
selected_features = [col for col in selected_features if col in current_df.columns]
print(f"Features available in the chosen dataset: {selected_features}")

if not selected_features:
     raise ValueError("No features selected or available in the chosen dataset. Stopping.")

y = current_df[target_variable] # Target
X = current_df[selected_features] # Features


# --- Train-Test Split ---
print("\n--- Splitting Data into Train/Test Sets ---")
test_set_size = 0.20

# Check if stratification is possible
if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2:
    stratify_option = y
    print(f"  Splitting with stratification based on '{target_variable}'.")
else:
    stratify_option = None
    print(f"  Warning: Cannot stratify split. Target variable '{target_variable}' might have only one class or too few samples in one class.")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_set_size,
    random_state=random_seed,
    stratify=stratify_option
)
print(f"  Train set shape: X={X_train.shape}, y={y_train.shape}")
print(f"  Test set shape: X={X_test.shape}, y={y_test.shape}")
print(f"  Test set '{target_variable}' distribution:\n{y_test.value_counts(normalize=True)}")


# --- START: Preprocessing Before Models (Missing Values, Scaling) ---

#Handle Missing Values
print("\n--- Handling Missing Values ---")
# Check if X_train/X_test are empty
if X_train.empty or X_test.empty:
     raise ValueError("X_train or X_test is empty after feature selection/split.")

#Check if numeric
numeric_cols_train = X_train.select_dtypes(include=np.number).columns
if len(numeric_cols_train) != X_train.shape[1]:
    non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Warning: Non-numeric columns detected in X_train after feature selection: {non_numeric_cols}. Attempting to drop them.")
    X_train = X_train.drop(columns=non_numeric_cols)
    X_test = X_test.drop(columns=non_numeric_cols)
    numeric_cols_train = X_train.columns # Update numeric cols list
    print(f"  Non-numeric columns dropped. Remaining features: {numeric_cols_train.tolist()}")


if X_train.isnull().sum().sum() > 0:
    print(f"  Found {X_train.isnull().sum().sum()} missing values in X_train. Imputing with median.")
    train_medians = X_train[numeric_cols_train].median()
    # Fill NaN in training set with training medians
    X_train.fillna(train_medians, inplace=True)
    # Fill NaN in test set with training medians (to avoid data leakage)
    X_test.fillna(train_medians, inplace=True)
    print("  Missing values imputed using training set medians.")
else:
    print("  No missing values found in X_train.")


# 2. Scaling Features
print("\n--- Scaling Features ---")
print("Apply scaling? (y / n) (Recommended for KNN, NB, DL)")
ans = input()
if ans.lower() == 'y':
    print("  Applying StandardScaler.")
    scaler = StandardScaler()
    # Fit scaler ONLY on the training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform both training and test data
    X_test_scaled = scaler.transform(X_test)
    # Convert back to DataFrame for compatibility if needed, though numpy arrays are fine for models
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

else:
     print("  Skipping feature scaling.")
     X_train_scaled = X_train.copy() # Use copy to avoid potential SettingWithCopyWarning later
     X_test_scaled = X_test.copy()


# --- KNN Model ---
if not X_train_scaled.empty and not y_train.empty:
    print(f"\n--- K-Nearest Neighbors (KNN) Classifier (Predicting '{target_variable}') ---")

    k_value = 6
    knn_model = KNeighborsClassifier(n_neighbors=k_value)

    knn_model.fit(X_train_scaled, y_train) # Train on scaled data

    y_pred_knn = knn_model.predict(X_test_scaled) # Predict on scaled data

    print("\n--- Evaluating KNN Model ---")
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Model Accuracy: {accuracy_knn:.4f}")
    print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_knn, zero_division=0))
    print("\nKNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


# --- Naive Bayes Model ---
if not X_train_scaled.empty and not y_train.empty:
    print(f"\n--- Gaussian Naive Bayes Classifier (Predicting '{target_variable}') ---")

    nb_model = GaussianNB()

    nb_model.fit(X_train_scaled, y_train) # Train on scaled data

    y_pred_nb = nb_model.predict(X_test_scaled) # Predict on scaled data

    print("\n--- Evaluating Naive Bayes Model ---")
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Model Accuracy: {accuracy_nb:.4f}")
    print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb, zero_division=0))
    print("\nNaive Bayes Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))


# --- Decision Tree Model ---
# Decision Trees are less sensitive to scaling, but using scaled data is fine
if not X_train_scaled.empty and not y_train.empty:
    print(f"\n--- Decision Tree Classifier (Predicting '{target_variable}') ---")

    # Using class_weight='balanced' can help with imbalanced datasets if undersampling wasn't used or wasn't perfect
    dt_model = DecisionTreeClassifier(random_state=random_seed, class_weight='balanced')

    dt_model.fit(X_train_scaled, y_train) # Train on scaled data

    y_pred_dt = dt_model.predict(X_test_scaled) # Predict on scaled data

    print("\n--- Evaluating Decision Tree Model ---")
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Model Accuracy: {accuracy_dt:.4f}")
    print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))
    print("\nDecision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))


# --- Random Forest Model ---
# Random Forests are also less sensitive to scaling
if not X_train_scaled.empty and not y_train.empty:
    print(f"\n--- Random Forest Classifier (Predicting '{target_variable}') ---")

    # Using class_weight='balanced' can help
    rf_model = RandomForestClassifier(random_state=random_seed, class_weight='balanced', n_estimators=100) # Added n_estimators

    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_test_scaled)

    print("\n--- Evaluating Random Forest Model ---")
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Model Accuracy: {accuracy_rf:.4f}")
    print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))
    print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# --- Deep Learning Model (MLP) ---
if not X_train_scaled.empty and not y_train.empty:
    print(f"\n--- Deep Learning (MLP) Classifier (Predicting '{target_variable}') ---")

    # Define the model architecture
    n_features = X_train_scaled.shape[1]
    dl_model = Sequential([
        Dense(64, activation='relu', input_shape=(n_features,)), # Input layer + first hidden layer
        Dropout(0.3), # Dropout for regularization
        Dense(32, activation='relu'), # Second hidden layer
        Dropout(0.2), # Dropout
        Dense(1, activation='sigmoid') # Output layer for binary classification
    ])

    # Compile the model
    dl_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

    print("\n--- Training Deep Learning Model ---")
    # Optional: Early stopping to prevent overfitting and save time
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    # Ensure y_train and y_test are numpy arrays for TensorFlow
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    history = dl_model.fit(X_train_scaled, y_train_np,
                           epochs=100, # Number of training cycles
                           batch_size=32, # Number of samples per gradient update
                           validation_data=(X_test_scaled, y_test_np),
                           callbacks=[early_stopping], # Add early stopping
                           verbose=0) # Set verbose=1 or 2 to see training progress per epoch

    print("Deep Learning Model training complete.")
    print(f"Training stopped after {len(history.history['loss'])} epochs.")


    print("\n--- Evaluating Deep Learning Model ---")
    # Evaluate the model on the test set
    loss, accuracy_dl = dl_model.evaluate(X_test_scaled, y_test_np, verbose=0)
    print(f"Deep Learning Model Accuracy: {accuracy_dl:.4f}")
    print(f"Deep Learning Model Loss: {loss:.4f}")


    # Make predictions (outputs probabilities)
    y_pred_prob_dl = dl_model.predict(X_test_scaled)
    # Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold
    y_pred_dl = (y_pred_prob_dl > 0.5).astype("int32").flatten() # Use flatten() to make it 1D array

    print("\nDeep Learning Classification Report:\n", classification_report(y_test, y_pred_dl, zero_division=0))
    print("\nDeep Learning Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dl))


# --- END: Model Code ---
print("\n--- Script Finished ---")
