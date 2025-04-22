import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# Import necessary metrics, including f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- Load Data ---
# Consider using a larger dataset if available for a 3-way split,
# but the code will work with the existing one.
df = pd.read_csv('employee_attrition_dataset_10000.csv')
print("Dataset loaded.")

# --- PREPARATION ---

df_prepared = df.drop('Employee_ID', axis=1)

# --- Label Encoding ---
columns_to_encode = ['Attrition', 'Gender']
label_encoder_attrition = LabelEncoder()
label_encoder_gender = LabelEncoder()

print("--- Label Encoding ---")
# Encode Attrition
original_values_attrition = df_prepared['Attrition'].unique()
df_prepared['Attrition'] = label_encoder_attrition.fit_transform(df_prepared['Attrition'])
encoded_values_attrition = df_prepared['Attrition'].unique()
mapping_attrition = dict(zip(label_encoder_attrition.classes_, label_encoder_attrition.transform(label_encoder_attrition.classes_)))
print(f"  Encoded 'Attrition': Original {original_values_attrition} -> Encoded {encoded_values_attrition}. Mapping: {mapping_attrition}")


# Encode Gender
original_values_gender = df_prepared['Gender'].unique()
df_prepared['Gender'] = label_encoder_gender.fit_transform(df_prepared['Gender'])
encoded_values_gender = df_prepared['Gender'].unique()
mapping_gender = dict(zip(label_encoder_gender.classes_, label_encoder_gender.transform(label_encoder_gender.classes_)))
print(f"  Encoded 'Gender': Original {original_values_gender} -> Encoded {encoded_values_gender}. Mapping: {mapping_gender}")


# Rename the encoded column to Gender_Male IF Male is encoded as 1
if 'Male' in mapping_gender and mapping_gender['Male'] == 1:
        df_prepared.rename(columns={'Gender': 'Gender_Male'}, inplace=True)
        print("  Renamed encoded 'Gender' column to 'Gender_Male'.")
elif 'Female' in mapping_gender and mapping_gender['Female'] == 1:
        # If Female is 1, we need to flip the bits to get Gender_Male (where Male=1)
        print("  Adjusting 'Gender' encoding to create 'Gender_Male' (Male=1).")
        df_prepared['Gender_Male'] = 1 - df_prepared['Gender']
        df_prepared.drop('Gender', axis=1, inplace=True)
else:
     print("  'Gender' column encoding did not result in 'Gender_Male' (Male=1). Check mapping.")


# --- One-Hot Encoding ---
print("\n--- One-Hot Encoding ---")
columns_to_encode_onehot = ['Department'] # Job_Role, Marital_Status removed
columns_exist_for_onehot = [col for col in columns_to_encode_onehot if col in df_prepared.columns]

if columns_exist_for_onehot:
    df_prepared = pd.get_dummies(df_prepared, columns=columns_exist_for_onehot, drop_first=False)
    print(f"  One-Hot Encoding applied to: {columns_exist_for_onehot}")
else:
    print("  No columns found for One-Hot Encoding from the specified list.")


# --- Undersampling Option ---
print("\n--- Undersampling Option ---")
print("Apply undersampling? (y / n)")
apply_undersampling = input().strip().lower() == 'y'
df_model_input = None # Will hold the data used for splitting

if apply_undersampling:
    print("\n--- Performing Undersampling ---")
    random_seed = 42

    # --- CONTROL PARAMETER ---
    undersampling_ratio = 1.0 # Balance: 1.0 means 1:1 ratio

    # Separate majority and minority classes (Attrition: No=0, Yes=1)
    df_majority = df_prepared[df_prepared.Attrition == 0]
    df_minority = df_prepared[df_prepared.Attrition == 1]

    minority_size = len(df_minority)
    majority_size = len(df_majority)

    print(f"Original distribution: Majority (0)={majority_size}, Minority (1)={minority_size}")
    print(f"Target undersampling ratio (Majority:Minority): {undersampling_ratio}:1")

    if majority_size < minority_size:
         print("Warning: Initial 'majority' class (Attrition=0) is smaller than 'minority' class (Attrition=1). Undersampling may not be effective.")
         n_samples_majority = majority_size # Keep all majority samples
    elif minority_size == 0:
        print("Error: Minority class size is zero. Cannot perform undersampling.")
        # Handle this case appropriately, maybe exit or skip undersampling
        n_samples_majority = majority_size # Default to keeping all majority
        df_model_input = df_prepared # Use original data if undersampling fails
    else:
         # Calculate the desired number of majority samples
         desired_majority_samples = int(minority_size * undersampling_ratio)
         n_samples_majority = min(desired_majority_samples, majority_size) # Ensure we don't exceed available samples

    print(f"Calculated samples to keep from majority class: {n_samples_majority}")

    if minority_size > 0:
        # Undersample the majority class
        df_majority_undersampled = df_majority.sample(n=n_samples_majority, random_state=random_seed)
        # Combine with the minority class
        df_undersampled = pd.concat([df_majority_undersampled, df_minority])
        # Shuffle the resulting DataFrame
        df_model_input = df_undersampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        print("Final dataset 'Attrition' distribution after undersampling:\n", df_model_input.Attrition.value_counts())
        if len(df_minority) > 0:
             final_ratio = len(df_model_input[df_model_input.Attrition == 0]) / len(df_model_input[df_model_input.Attrition == 1])
             print(f"Final ratio (Majority/Minority): {final_ratio:.2f}:1")
        else:
             print("Final ratio calculation skipped as minority size is zero.")
    else:
        # If minority size was 0, df_model_input might still be None or df_prepared
        if df_model_input is None:
             df_model_input = df_prepared
        print("Skipped undersampling combination due to zero minority samples.")

else:
    print("\n--- Skipping Undersampling ---")
    df_model_input = df_prepared.reset_index(drop=True) # Use the prepared data directly

# --- Feature Selection (Select Specific Features) ---
print("\n--- Feature Selection (Selecting Specific Features) ---")
# Define the list of features we want to use for modeling.
# Ensure these columns exist AFTER encoding and potential renaming (e.g., Gender_Male)
selected_features = [
    'Gender_Male',
    'Department_IT', # Example one-hot encoded feature
    # Add other one-hot encoded department columns if they exist and are desired
    'Department_HR',
    'Department_Sales',
    'Age',
    'Years_Since_Last_Promotion',
    'Work_Life_Balance',
    'Performance_Rating',
    'Training_Hours_Last_Year',
    'Average_Hours_Worked_Per_Week',
    'Absenteeism',
    'Job_Involvement'
]

# Filter selected_features to only include columns present in the current dataframe
available_features = [col for col in selected_features if col in df_model_input.columns]
missing_features = [col for col in selected_features if col not in df_model_input.columns]

if missing_features:
    print(f"Warning: The following selected features were not found in the dataframe: {missing_features}")
print(f"Using the following available features for modeling: {available_features}")

# --- Define Target (y) and Features (X) ---
print("\n--- Defining Target (y) and Features (X) ---")
target_variable = 'Attrition'

if target_variable not in df_model_input.columns:
    raise ValueError(f"Target variable '{target_variable}' not found in the dataframe.")
if not available_features:
     raise ValueError("No features selected or available for modeling.")

y = df_model_input[target_variable]
X = df_model_input[available_features]

print(f"Target variable (y): {target_variable}")
print("Shape of X before split:", X.shape)
print("Shape of y before split:", y.shape)
if X.empty or y.empty:
    raise ValueError("X or y is empty before splitting. Check data loading and preparation steps.")


# --- Train-Validation-Test Split ---
print("\n--- Splitting Data into Train/Validation/Test Sets (60:20:20) ---")
random_seed = 42
test_set_size = 0.20 # Test set size (20%)
validation_set_size_relative = 0.25 # Validation set size relative to the remaining data (0.25 * 0.80 = 0.20)

# Step 1: Split into Training+Validation (80%) and Test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=test_set_size,
    random_state=random_seed,
    stratify=y # Stratify based on the target variable
)

# Step 2: Split Training+Validation into Training (60% of total) and Validation (20% of total)
# The new test_size here is validation_set_size_relative of the train_val set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=validation_set_size_relative,
    random_state=random_seed, # Use the same random state for reproducibility
    stratify=y_train_val # Stratify based on the train_val target
)

print("\n--- Verification of Train/Validation/Test Split ---")
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of y_test:", y_test.shape)

# Verify distributions (optional but recommended)
print(f"\n'{target_variable}' distribution in original y:\n", y.value_counts(normalize=True))
print(f"\n'{target_variable}' distribution in y_train:\n", y_train.value_counts(normalize=True))
print(f"\n'{target_variable}' distribution in y_val:\n", y_val.value_counts(normalize=True))
print(f"\n'{target_variable}' distribution in y_test:\n", y_test.value_counts(normalize=True))


# --- Preprocessing Before Models (Missing Values, Scaling) ---

# 1. Handle Missing Values
print("\n--- Handling Missing Values ---")
# Check if X_train/X_val/X_test are empty
if X_train.empty or X_val.empty or X_test.empty:
     raise ValueError("X_train, X_val, or X_test is empty after splitting.")

# Check if numeric (should be, based on feature selection, but good practice)
numeric_cols_train = X_train.select_dtypes(include=np.number).columns
if len(numeric_cols_train) != X_train.shape[1]:
    non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Warning: Non-numeric columns detected in X_train after feature selection: {non_numeric_cols}. This might cause issues.")
    # Consider adding explicit handling or ensure only numeric features are selected

# Calculate medians ONLY from the training set
train_medians = X_train[numeric_cols_train].median()
print(f"Calculated medians from Training set: \n{train_medians}")

# Fill NaN in training, validation, and test sets using TRAINING medians
X_train.fillna(train_medians, inplace=True)
X_val.fillna(train_medians, inplace=True)
X_test.fillna(train_medians, inplace=True)
print("Missing values filled using training set medians.")

# 2. Scaling Features
print("\n--- Scaling Features ---")
print("Apply scaling? (y / n)")
apply_scaling = input().strip().lower() == 'y'

X_train_processed = X_train.copy() # Start with unscaled data
X_val_processed = X_val.copy()
X_test_processed = X_test.copy()

if apply_scaling:
    scaler = StandardScaler()
    # Fit scaler ONLY on the training data
    print("Fitting StandardScaler on Training data...")
    scaler.fit(X_train)
    # Transform training, validation, and test data
    X_train_processed = scaler.transform(X_train)
    X_val_processed = scaler.transform(X_val)
    X_test_processed = scaler.transform(X_test)
    print("Features scaled using StandardScaler.")
    # Convert back to DataFrame to preserve column names if needed later (optional)
    # X_train_processed = pd.DataFrame(X_train_processed, columns=X_train.columns, index=X_train.index)
    # X_val_processed = pd.DataFrame(X_val_processed, columns=X_val.columns, index=X_val.index)
    # X_test_processed = pd.DataFrame(X_test_processed, columns=X_test.columns, index=X_test.index)
else:
     print("Skipping feature scaling.")
     # X_train_processed, X_val_processed, X_test_processed remain as the original filled dataframes


# --- KNN Model with Hyperparameter Tuning ---
print(f"\n--- K-Nearest Neighbors (KNN) Classifier ---")

# Define range of k values to test
k_values = [3, 5, 7, 9, 11, 13, 15] # Example range
best_k = -1
best_f1_score = -1

print(f"Tuning KNN: Testing k values {k_values} using the Validation set.")
print(f"Performance Measure: Weighted F1-Score")

for k in k_values:
    # 1. Train on the TRAINING set
    knn_model_val = KNeighborsClassifier(n_neighbors=k)
    knn_model_val.fit(X_train_processed, y_train) # Use processed (potentially scaled) data

    # 2. Evaluate on the VALIDATION set
    y_pred_val_knn = knn_model_val.predict(X_val_processed) # Predict on processed validation data
    current_f1_score = f1_score(y_val, y_pred_val_knn, average='weighted', zero_division=0) # Use weighted F1 for potential imbalance
    print(f"  k={k}: Validation Weighted F1-Score = {current_f1_score:.4f}")

    # 3. Keep track of the best k
    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        best_k = k

print(f"\nBest k found: {best_k} (Validation Weighted F1-Score: {best_f1_score:.4f})")

# 4. Train the FINAL KNN model using the best k on the TRAINING set
print(f"\nTraining final KNN model with k={best_k} on the Training set...")
final_knn_model = KNeighborsClassifier(n_neighbors=best_k)
final_knn_model.fit(X_train_processed, y_train)

# 5. Evaluate the FINAL model on the TEST set
print("\n--- Evaluating FINAL KNN Model on the **Test Set** ---")
y_pred_test_knn = final_knn_model.predict(X_test_processed) # Predict on processed test data

accuracy_knn_test = accuracy_score(y_test, y_pred_test_knn)
print(f"KNN Final Test Accuracy: {accuracy_knn_test:.4f}")
print("\nKNN Final Test Classification Report:\n", classification_report(y_test, y_pred_test_knn, zero_division=0))
print("\nKNN Final Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_knn))


# --- Naive Bayes Model ---
# No hyperparameter tuning in this example, so train on Train, evaluate on Test
print(f"\n--- Gaussian Naive Bayes Classifier ---")

nb_model = GaussianNB()

# Train on the TRAINING set
print("Training Naive Bayes model on the Training set...")
nb_model.fit(X_train_processed, y_train) # Use processed (potentially scaled) data

# Evaluate on the TEST set
print("\n--- Evaluating Naive Bayes Model on the **Test Set** ---")
y_pred_test_nb = nb_model.predict(X_test_processed) # Predict on processed test data

accuracy_nb_test = accuracy_score(y_test, y_pred_test_nb)
print(f"Naive Bayes Test Accuracy: {accuracy_nb_test:.4f}")
print("\nNaive Bayes Test Classification Report:\n", classification_report(y_test, y_pred_test_nb, zero_division=0))
print("\nNaive Bayes Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_nb))


# --- Decision Tree Model ---
# No hyperparameter tuning in this example, so train on Train, evaluate on Test
# Note: Could add tuning for 'max_depth', 'min_samples_split' etc. using the validation set
print(f"\n--- Decision Tree Classifier ---")

# Added class_weight='balanced' as before, useful if undersampling wasn't used or perfect
dt_model = DecisionTreeClassifier(random_state=random_seed, class_weight='balanced')

# Train on the TRAINING set
print("Training Decision Tree model on the Training set...")
dt_model.fit(X_train_processed, y_train) # Use processed (potentially scaled) data

# Evaluate on the TEST set
print("\n--- Evaluating Decision Tree Model on the **Test Set** ---")
y_pred_test_dt = dt_model.predict(X_test_processed) # Predict on processed test data

accuracy_dt_test = accuracy_score(y_test, y_pred_test_dt)
print(f"Decision Tree Test Accuracy: {accuracy_dt_test:.4f}")
print("\nDecision Tree Test Classification Report:\n", classification_report(y_test, y_pred_test_dt, zero_division=0))
print("\nDecision Tree Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_dt))


# --- Random Forest Model ---
# No hyperparameter tuning in this example, so train on Train, evaluate on Test
# Note: Could add tuning for 'n_estimators', 'max_depth' etc. using the validation set
print(f"\n--- Random Forest Classifier ---")

# Added class_weight='balanced' as before
rf_model = RandomForestClassifier(random_state=random_seed, class_weight='balanced')

# Train on the TRAINING set
print("Training Random Forest model on the Training set...")
rf_model.fit(X_train_processed, y_train) # Use processed (potentially scaled) data

# Evaluate on the TEST set
print("\n--- Evaluating Random Forest Model on the **Test Set** ---")
y_pred_test_rf = rf_model.predict(X_test_processed) # Predict on processed test data

accuracy_rf_test = accuracy_score(y_test, y_pred_test_rf)
print(f"Random Forest Test Accuracy: {accuracy_rf_test:.4f}")
print("\nRandom Forest Test Classification Report:\n", classification_report(y_test, y_pred_test_rf, zero_division=0))
print("\nRandom Forest Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_rf))

# --- END: Model Code ---
print("\nScript finished.")
