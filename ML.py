import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

df = pd.read_csv('employee_attrition_dataset.csv')

#PREPARATION

df_prepared = df.drop('Employee_ID', axis=1)

#label-encode
columns_to_encode = ['Attrition', 'Overtime']

label_encoder = LabelEncoder()

for col in columns_to_encode:
    # Store original values before encoding for verification
    original_values = df_prepared[col].unique()
    # Fit the encoder and transform the column
    df_prepared[col] = label_encoder.fit_transform(df_prepared[col])
    # Print mapping for clarity
    encoded_values = df_prepared[col].unique()
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"  Encoded '{col}': Original unique values {original_values} -> Encoded unique values {encoded_values}. Mapping: {mapping}")

#one-hot-encoding
columns_to_encode_onehot = ['Marital_Status', 'Department']
columns_exist_for_onehot = [col for col in columns_to_encode_onehot if col in df_prepared.columns]

df_prepared = pd.get_dummies(df_prepared, columns=columns_exist_for_onehot, drop_first=False) 
print("  One-Hot Encoding applied.")
print("\nColumns after One-Hot Encoding:", df_prepared.columns.tolist())

#Remove useless features

low_mi_features_to_drop = [
    'Work_Life_Balance',            # MI = 0.000000
    'Work_Environment_Satisfaction',# MI = 0.000280
    'Absenteeism',                  # MI = 0.000358
    'Gender',                       # MI = 0.001705 (Label Encoded)
    'Average_Hours_Worked_Per_Week', # MI = 0.002019
    'Job_Role'
    # Add more low-scoring features here if desired
]
features_to_drop = ['Attrition'] + low_mi_features_to_drop

# Ensure the columns actually exist in df_prepared before trying to drop
features_to_drop = [col for col in features_to_drop if col in df_prepared.columns]
print(f"\nDropping features to create X: {features_to_drop}")

# Splitting the data
y = df_prepared['Attrition'] # Label
X = df_prepared.drop(columns=features_to_drop) # Features - Dropping the specified list

#Split into test and training

test_set_size = 0.20
random_seed = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_set_size,
    random_state=random_seed,
    stratify=y  
)

print("\n--- Verification of Train/Test Split ---")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

print("\nAttrition distribution in original y:\n", y.value_counts(normalize=True))
print("\nAttrition distribution in y_train:\n", y_train.value_counts(normalize=True))
print("\nAttrition distribution in y_test:\n", y_test.value_counts(normalize=True))

#KNN
if X_train is not None and y_train is not None:

    # print("\n--- Scaling Features ---")
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test) 
    # print("Features scaled using StandardScaler.")

    k_value = 6
    print(f"Initializing KNN classifier with k={k_value}")
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    print("Fitting the KNN model to the training data...")
    knn_model.fit(X_train, y_train) 

    y_pred = knn_model.predict(X_test)
    print("Predictions made on the test set.")

    print("\n--- Evaluating Model (Example) ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

