import numpy as np
from mpl_toolkits.axes_grid1.axes_size import AxesX
from pandas import read_csv
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = read_csv('employee_attrition_dataset.csv') #read the CSV

test_flag = False
if test_flag:
    (dataset.head(12)) #To print the first 12 rows

test_type_print = False
if test_type_print:
    print(dataset.tail(12)) #To print the last 12 rows

test_type_print = False
if test_type_print:
    print(dataset.dtypes) #To print the data types for each column

test_firstcolumnname = False
if test_firstcolumnname:
    print(dataset.columns[0]) #To print the first column name

test_type_print = False
if test_type_print:
    print(dataset.info()) #To print the data types for each column

test_distinct_vals = False
if test_distinct_vals:
    num_distinct_values = dataset['Age'].nunique() #Choose a categorical attribute and display the distinct values it contains
    distinct_values = dataset['Age'].unique()
    print("\nNumber of distinct values in 'Age': {}".format(num_distinct_values))
    print("Distinct values in 'Age':")
    print(distinct_values)

test_frequent_value = False
if test_frequent_value:
    print(dataset['Gender'].mode()) #To print the most frequently occurring value in the chosen categorical attribute

test_mean_median = False
if test_mean_median:
    column = 'Age' #To calculate the mean, median, standard deviation and the quantile for the select column
    print(dataset[column].mean())
    print(dataset[column].median())
    print(dataset[column].std())
    print(dataset[column].quantile(0.20))

test_filter_attribute = False
if test_filter_attribute:
    filtered_data = dataset[dataset['Age'] > 30] #Filter based on age
    print("Filtered data (Age > 30):")
    print(filtered_data)

test_filter_letter = False
if test_filter_letter:
    filtered_on_name = dataset[dataset['Gender'].str.startswith('F')] #To filter based on letter
    print(filtered_on_name)

test_duplicate_remove = False
if test_duplicate_remove:
    no_duplicates = dataset.drop_duplicates() #To remove the dupilcated rows
    print (no_duplicates)

test_type_change = False
if test_type_change:
    dataset['Age'] = dataset['Age'].astype(str) #To change the data type from int to str
    print(dataset.dtypes)

test_group = False
if test_group:
    grouped_data = dataset.groupby(['Gender', 'Marital_Status']).size().reset_index(name='Count') #Group data based on two attributes
    print(grouped_data)

test_check_missing = False
if test_check_missing:
    missing_values = dataset.isnull().sum() #Check for missing vals
    print("Missing values in the dataset:")
    print(missing_values)

test_missing_vals = False
if test_missing_vals:
    missing_values = dataset.isnull().sum() #Check for missing vals
    print("Missing values in the dataset:")
    print(missing_values)

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median()) #Replace missing values
    dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode()[0])
    print("\nMissing values replaced.")

    missing_values = dataset.isnull().sum()  # Re-Check for missing vals
    print("Missing values in the dataset:")
    print(missing_values)

test_bin = False
if test_bin:
    dataset['Monthly_Income_bins'] = pd.cut(dataset['Monthly_Income'], bins=5) #Divide in bins and count
    bin_counts = dataset['Monthly_Income_bins'].value_counts().sort_index()
    print(bin_counts)

test_maxrow = False
if test_maxrow:
    max_row = dataset.loc[dataset['Monthly_Income'].idxmax()] #To find the row with the maximum value
    print(max_row)

test_boxplot = False
if test_boxplot:
    sns.boxplot(y=dataset['Monthly_Income']) #boxplot for Monthly Income since it's significant for Employee's satisfaction
    plt.title('Boxplot of Monthly Income')
    plt.ylabel('Monthly Income')
    plt.show()
    #Median at 12000$

test_histplot = False
if test_histplot:
    sns.histplot(x = dataset['Monthly_Income']) #histogram for Monthly Income
    plt.title('Histogram of Monthly Income')
    plt.xlabel('Monthly Income')
    plt.show()
    #The graph is normal bell shape, 20000 is the mode. Data is variable with no outliers

test_scatter = False
if test_scatter:
    plt.scatter(x = dataset['Monthly_Income'], y = dataset['Job_Satisfaction']) #scatter plot for Monthly Income and Job Satisfaction
    plt.title('Scatter Plot of Monthly Income and Job Satisfaction')
    plt.xlabel('Monthly Income')
    plt.ylabel('Job Satisfaction')
    plt.show()
    #This shows very weak correlation between monthly income and job satisfaction

test_normalization = False
if test_normalization:
    numerical_features = dataset.select_dtypes(include=['number']) #Normalize numerical attributes
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features.columns)
    print(scaled_numerical_df.head().to_markdown(index=False, numalign="left", stralign="left"))

test_PCA = False
if test_PCA:
    #Select columns with numerical data
    numerical_features = dataset.select_dtypes(include=['number'])
    # replace missing values with the median of each column
    numerical_features = numerical_features.fillna(numerical_features.median())
    # Standardize the numerical features
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
    # Apply PCA with 2 components
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_numerical)
    # Create a DataFrame for the principal components
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    # Display the rows
    #print(principalDf.head().to_markdown(index=False, numalign="left", stralign="left"))
    # Print the column names and their data types
    #print(principalDf.info())
    # Visualize the first two standardized numerical features
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(scaled_numerical[:, 0], scaled_numerical[:, 1])
    plt.title('Data Before PCA')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # Visualize the principal components
    plt.subplot(1, 2, 2)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.title('Data After PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # Show the plots
    plt.show()

test_heatmap = False
if test_heatmap:
    numerical_features = dataset.select_dtypes(include=['number'])
    correlation_matrix = numerical_features.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()
    #Strong correlation between years at company and getting a promotion (0.7)

test_pA = True #Analytics part A
if test_pA:
    numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
    #Use this var to get for all numerical attributes ^
    correlation_matrix = dataset[['Monthly_Income', 'Job_Satisfaction', 'Distance_From_Home', 'Years_at_Company']].corr()
    # from 1 (meaning positively correlated) to -1 (meaning negatively correlated)
    print("Correlation Matrix:")
    print(correlation_matrix)

test_pB = False #Analytics part B
if test_pB:
    dataset = pd.read_csv('employee_attrition_dataset.csv') #Check balance of variables
    variable_count = dataset['Gender'].value_counts()
    print("Number of both variables respectively:")
    print(variable_count)
    print("Percentage of both variables: ")
    print(variable_count / len(dataset) * 100)
    #For gender dataset is balanced

test_pC = False   #Analytics part C
if test_pC:
    dataset['Income_Satisfaction'] = dataset['Monthly_Income'] * dataset['Job_Satisfaction']
    # Create new feature to combine income and satisfaction, drawing a connection
    dataset['Age_Squared'] = dataset['Age'] ** 2
    #Catches non-linear relationships between Age and other features
    dataset['Income_Bin'] = pd.cut(dataset['Monthly_Income'], bins=[3000, 6000, 8000, 20000], labels=['Low', 'Medium', 'High'])
    #Categorize income into bin with low, medium and high
    dataset['Income_to_Age_Ratio'] = dataset['Monthly_Income'] / dataset['Age']
    #Shows income relatively to age
    print("Dataset with New Features:")
    new_features = ['Income_Satisfaction', 'Age_Squared', 'Income_Bin', 'Income_to_Age_Ratio']
    print(dataset[new_features])


    #Correlation_test

# Calculate mutual information scores (requires X_train, y_train)
# Ensure data is numerical (which it should be after encoding)
# mi_scores = mutual_info_classif(X_train, y_train, random_state=random_seed)
# mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train.columns)
# mi_scores = mi_scores.sort_values(ascending=False)

# print("Mutual Information Scores (Top 15):")
# print(mi_scores.head(15))

# # Plotting MI scores
# plt.figure(figsize=(10, 6))
# mi_scores.head(15).plot(kind='barh') # Plot top 15
# plt.title('Top 15 Features by Mutual Information Score')
# plt.xlabel('MI Score')
# plt.show()

# You can then select the top N features based on these scores
# top_n = 15
# selected_features_mi = mi_scores.head(top_n).index.tolist()
# X_train_selected = X_train[selected_features_mi]
# X_test_selected = X_test[selected_features_mi]

#Cleanup?

# # Example: Remove features with zero variance (constant features)
# selector = VarianceThreshold(threshold=0.0)
# # Fit on training data (or all X if done before split)
# selector.fit(X_train)
# # Get boolean mask of features to keep
# mask = selector.get_support()
# # Apply mask to get selected features
# X_train_selected = X_train.loc[:, mask]
# X_test_selected = X_test.loc[:, mask] # Use same mask for test set

# print(f"Original feature count: {X_train.shape[1]}")
# print(f"Features after Variance Threshold: {X_train_selected.shape[1]}")
# Now use X_train_selected, X_test_selected for scaling and modeling
