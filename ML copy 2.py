import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Feature selection imports removed as they are not used after preprocessing in this version
# import matplotlib.pyplot as plt # Keep if you plan plots later
from sklearn.naive_bayes import GaussianNB
# ADDED: Imports for Decision Tree and Random Forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# REMOVED: shuffle import no longer needed for sampling method
# from sklearn.utils import shuffle

df = pd.read_csv('employee_attrition_dataset_10000.csv')

# --- PREPARATION ---

df_prepared = df.drop('Employee_ID', axis=1)

# --- Label Encoding ---
columns_to_encode = ['Attrition', 'Overtime']
label_encoder = LabelEncoder()
print("--- Label Encoding ---")
for col in columns_to_encode:
    original_values = df_prepared[col].unique()
    df_prepared[col] = label_encoder.fit_transform(df_prepared[col])
    encoded_values = df_prepared[col].unique()
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"  Encoded '{col}': Original unique values {original_values} -> Encoded unique values {encoded_values}. Mapping: {mapping}")


# --- One-Hot Encoding ---
print("\n--- One-Hot Encoding ---")
columns_to_encode_onehot = ['Marital_Status', 'Department']
columns_exist_for_onehot = [col for col in columns_to_encode_onehot if col in df_prepared.columns]
df_prepared = pd.get_dummies(df_prepared, columns=columns_exist_for_onehot, drop_first=False)
print("  One-Hot Encoding applied.")
print("Columns after One-Hot Encoding:", df_prepared.columns.tolist())

# --- START: Undersampling Implementation ---
print("\n--- Performing Random Undersampling ---")
random_seed = 42 # Use the same random seed for reproducibility

# Separate majority and minority classes (assuming Attrition 'Yes' is 1 and 'No' is 0)
df_majority = df_prepared[df_prepared.Attrition == 0]
df_minority = df_prepared[df_prepared.Attrition == 1]

print(f"Original dataset shape: {df_prepared.shape}")
print(f"Majority class (Attrition=0) count: {len(df_majority)}")
print(f"Minority class (Attrition=1) count: {len(df_minority)}")

# Undersample the majority class
df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=random_seed)

# Combine minority class with undersampled majority class
df_undersampled = pd.concat([df_majority_undersampled, df_minority])

# Shuffle the resulting DataFrame
df_undersampled = df_undersampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)

print(f"\nUndersampled dataset shape: {df_undersampled.shape}")
print("Undersampled dataset 'Attrition' distribution:\n", df_undersampled.Attrition.value_counts())
# --- END: Undersampling Implementation ---


# --- Feature Selection (using the undersampled data) ---
print("\n--- Feature Selection ---")
# Define features to drop (same logic as before, but applied to df_undersampled)
low_mi_features_to_drop = [
    'Work_Life_Balance',
    'Work_Environment_Satisfaction',
    'Absenteeism',
    'Gender',
    'Average_Hours_Worked_Per_Week',
    'Job_Role'
]
# We need Attrition for y, so we define features_to_drop for X separately
features_to_drop_for_X = low_mi_features_to_drop # Job_Role was already dropped by one-hot if it existed

# Ensure the columns actually exist in df_undersampled before trying to drop
features_to_drop_for_X = [col for col in features_to_drop_for_X if col in df_undersampled.columns]
print(f"Dropping features to create X: {features_to_drop_for_X}")

# --- Splitting the UNDERSAMPLED data ---
print("\n--- Splitting Undersampled Data ---")
# Use df_undersampled instead of df_prepared
y = df_undersampled['Attrition'] # Label from undersampled data
# Drop 'Attrition' AND the low_mi_features to create X from undersampled data
X = df_undersampled.drop(columns=['Attrition'] + features_to_drop_for_X) # Features from undersampled data

print("Final features used for X:", X.columns.tolist()) # Verify the final features

# Split into test and training
test_set_size = 0.20
# random_seed is already defined above

X_train, X_test, y_train, y_test = train_test_split(
    X, # From undersampled data
    y, # From undersampled data
    test_size=test_set_size,
    random_state=random_seed,
    stratify=y # Stratify on the now balanced y
)

print("\n--- Verification of Train/Test Split (Undersampled Data) ---")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

print("\nAttrition distribution in original undersampled y:\n", y.value_counts(normalize=True))
print("\nAttrition distribution in y_train:\n", y_train.value_counts(normalize=True))
print("\nAttrition distribution in y_test:\n", y_test.value_counts(normalize=True))

# --- Scaling Features (for models that need it, like KNN) ---
# Note: Trees/Naive Bayes don't strictly require scaling, so we'll apply it selectively
print("\n--- Scaling Features ---")
scaler = StandardScaler()
# Fit scaler ONLY on the training data
X_train_scaled = scaler.fit_transform(X_train)
# Transform both training and test data
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler (will be used for KNN).")


# --- KNN Model (using SCALED undersampled data) ---
if X_train is not None and y_train is not None:
    print("\n--- K-Nearest Neighbors (KNN) Classifier (on Scaled Undersampled Data) ---")

    k_value = 6
    print(f"Initializing KNN classifier with k={k_value}")
    knn_model = KNeighborsClassifier(n_neighbors=k_value)

    print("Fitting the KNN model to the SCALED undersampled training data...")
    # MODIFIED: Fit on scaled data
    knn_model.fit(X_train_scaled, y_train)

    print("Predictions made on the SCALED test set using KNN.")
     # MODIFIED: Predict on scaled data
    y_pred_knn = knn_model.predict(X_test_scaled)

    print("\n--- Evaluating KNN Model (on Undersampled Data) ---")
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Model Accuracy: {accuracy_knn:.4f}")
    # Added zero_division=0
    print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_knn, zero_division=0))
    print("\nKNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


# --- Naive Bayes Model (using UNSCALED undersampled data) ---
# GaussianNB can technically handle scaled data, but often used unscaled.
# Let's keep it unscaled for consistency with the previous NB run.
if X_train is not None and y_train is not None:
    print("\n--- Gaussian Naive Bayes Classifier (on Unscaled Undersampled Data) ---")

    print("Initializing Gaussian Naive Bayes classifier...")
    nb_model = GaussianNB()

    print("Fitting the Naive Bayes model to the UNSCALED undersampled training data...")
    # Fit on unscaled data
    nb_model.fit(X_train, y_train)

    print("Predictions made on the UNSCALED test set using Naive Bayes.")
    # Predict on unscaled data
    y_pred_nb = nb_model.predict(X_test)

    print("\n--- Evaluating Naive Bayes Model (on Undersampled Data) ---")
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Model Accuracy: {accuracy_nb:.4f}")
    # Added zero_division=0
    print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb, zero_division=0))
    print("\nNaive Bayes Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))


# --- START: Added Decision Tree Model ---
if X_train is not None and y_train is not None:
    print("\n--- Decision Tree Classifier (on Unscaled Undersampled Data) ---")

    print("Initializing Decision Tree classifier...")
    # Using random_state for reproducibility
    dt_model = DecisionTreeClassifier(random_state=random_seed)

    print("Fitting the Decision Tree model to the UNSCALED undersampled training data...")
    # Trees don't require scaling
    dt_model.fit(X_train, y_train)

    print("Predictions made on the UNSCALED test set using Decision Tree.")
    # Predict on unscaled data
    y_pred_dt = dt_model.predict(X_test)

    print("\n--- Evaluating Decision Tree Model (on Undersampled Data) ---")
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Model Accuracy: {accuracy_dt:.4f}")
    # Added zero_division=0
    print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))
    print("\nDecision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
# --- END: Added Decision Tree Model ---


# --- START: Added Random Forest Model ---
if X_train is not None and y_train is not None:
    print("\n--- Random Forest Classifier (on Unscaled Undersampled Data) ---")

    print("Initializing Random Forest classifier...")
    # Using random_state for reproducibility, default n_estimators=100
    rf_model = RandomForestClassifier(random_state=random_seed)

    print("Fitting the Random Forest model to the UNSCALED undersampled training data...")
    # Forests also don't strictly require scaling
    rf_model.fit(X_train, y_train)

    print("Predictions made on the UNSCALED test set using Random Forest.")
    # Predict on unscaled data
    y_pred_rf = rf_model.predict(X_test)

    print("\n--- Evaluating Random Forest Model (on Undersampled Data) ---")
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Model Accuracy: {accuracy_rf:.4f}")
    # Added zero_division=0
    print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))
    print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
# --- END: Added Random Forest Model ---

# --- END: Model Code ---
