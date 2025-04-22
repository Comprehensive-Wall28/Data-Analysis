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
mapping = dict(zip(label_encoder_attrition.classes_, label_encoder_attrition.transform(label_encoder_attrition.classes_)))

# Encode Gender
original_values = df_prepared['Gender'].unique()
df_prepared['Gender'] = label_encoder_gender.fit_transform(df_prepared['Gender'])
encoded_values = df_prepared['Gender'].unique()
mapping = dict(zip(label_encoder_gender.classes_, label_encoder_gender.transform(label_encoder_gender.classes_)))

# Rename the encoded column to Gender_Male IF Male is encoded as 1
if 'Male' in mapping and mapping['Male'] == 1:
        df_prepared.rename(columns={'Gender': 'Gender_Male'}, inplace=True)
        print("  Renamed encoded 'Gender' column to 'Gender_Male'.")
elif 'Female' in mapping and mapping['Female'] == 1:
        # If Female is 1, we need to flip the bits to get Gender_Male (where Male=1)
        print("  Adjusting 'Gender' encoding to create 'Gender_Male' (Male=1).")
        df_prepared['Gender_Male'] = 1 - df_prepared['Gender']
        df_prepared.drop('Gender', axis=1, inplace=True) 


# --- One-Hot Encoding ---
print("\n--- One-Hot Encoding ---")
columns_to_encode_onehot = ['Department'] # Job_Role, Marital_Status removed
columns_exist_for_onehot = [col for col in columns_to_encode_onehot if col in df_prepared.columns]

df_prepared = pd.get_dummies(df_prepared, columns=columns_exist_for_onehot, drop_first=False)
print(f"  One-Hot Encoding applied to: {columns_exist_for_onehot}")


# ---Undersampling Implementation ---
print("\n--- Performing Undersampling ---")
random_seed = 42

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

# Combine the (potentially reduced) minority class with the undersampled majority class
df_undersampled = pd.concat([df_majority_undersampled, df_minority])

# Shuffle the resulting DataFrame
df_undersampled = df_undersampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)

print("Final dataset 'Attrition' distribution after undersampling:\n", df_undersampled.Attrition.value_counts())
print(f"Final ratio (Majority/Minority): {len(df_undersampled[df_undersampled.Attrition == 0]) / len(df_undersampled[df_undersampled.Attrition == 1]):.2f}:1")

# --- END: Undersampling Implementation ---

# --- START: Feature Selection (Select Specific Features) ---
print("\n--- Feature Selection (Selecting Specific Features) ---")

# Define the list of features we want to use for modeling.
selected_features = [
    'Gender_Male', 
    'Department_IT', 
    'Age',
    'Years_Since_Last_Promotion',
    'Work_Life_Balance',
    'Performance_Rating',
    'Training_Hours_Last_Year',
    'Average_Hours_Worked_Per_Week',
    'Absenteeism',
    'Job_Involvement'
]

# --- Define Target (y) and Features (X) from Undersampled Data ---
print("\n--- Defining Target (y) and Features (X) from Undersampled Data ---")
target_variable = 'Attrition' 
print("Undersample your data? (y / n)")
ans = input()
if ans.lower() == 'y':
    y = df_undersampled[target_variable] # Target 
    X = df_undersampled[selected_features] # Features 
else:
    y = df_prepared[target_variable] # Target 
    X = df_prepared[selected_features] # Features 


# --- Train-Test Split ---
print("\n--- Splitting Data into Train/Test Sets ---")
test_set_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=test_set_size,
    random_state=random_seed,
    stratify=y 
)

# --- START: Preprocessing Before Models (Missing Values, Scaling) ---

#Handle Missing Values
print("\n--- Handling Missing Values ---")
# Check if X_train/X_test are empty
if X_train.empty or X_test.empty:
     raise ValueError("X_train or X_test is empty after feature selection/split.")

#Check if numeric
numeric_cols_train = X_train.select_dtypes(include=np.number).columns
if len(numeric_cols_train) != X_train.shape[1]:
    print("Warning: Non-numeric columns detected in X_train after feature selection. This might cause issues.")
    print(f"Non-numeric columns: {X_train.select_dtypes(exclude=np.number).columns.tolist()}")

train_medians = X_train[numeric_cols_train].median()
# Fill NaN in training set with training medians
X_train.fillna(train_medians, inplace=True)
# Fill NaN in test set with training medians (to avoid data leakage)
X_test.fillna(train_medians, inplace=True)


# 2. Scaling Features
print("\n--- Scaling Features ---")
print("Apply scaling? (y / n)")
ans = input()
if ans.lower() == 'y':
    scaler = StandardScaler()
    # Fit scaler on the training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform both training and test data
    X_test_scaled = scaler.transform(X_test)
else:
     X_train_scaled = X_train
     X_test_scaled = X_test


# --- KNN Model ---
if X_train_scaled is not None and y_train is not None:
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
if X_train_scaled is not None and y_train is not None:
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
if X_train_scaled is not None and y_train is not None:
    print(f"\n--- Decision Tree Classifier (Predicting '{target_variable}') ---")

    dt_model = DecisionTreeClassifier(random_state=random_seed, class_weight='balanced')

    dt_model.fit(X_train_scaled, y_train) # Train on scaled data

    y_pred_dt = dt_model.predict(X_test_scaled) # Predict on scaled data

    print("\n--- Evaluating Decision Tree Model ---")
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Model Accuracy: {accuracy_dt:.4f}")
    print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))
    print("\nDecision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))


# --- Random Forest Model ---
if X_train_scaled is not None and y_train is not None:
    print(f"\n--- Random Forest Classifier (Predicting '{target_variable}') ---")

    rf_model = RandomForestClassifier(random_state=random_seed, class_weight='balanced')

    rf_model.fit(X_train_scaled, y_train) 

    y_pred_rf = rf_model.predict(X_test_scaled) 

    print("\n--- Evaluating Random Forest Model ---")
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Model Accuracy: {accuracy_rf:.4f}")
    print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))
    print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# --- END: Model Code ---
