import numpy as np
from mpl_toolkits.axes_grid1.axes_size import AxesX
from pandas import read_csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = read_csv('employee_attrition_dataset.csv') #read the CSV

test_flag = False #Enable to print all

if test_flag:

    (dataset.head(12)) #To print the first 12 rows

    print(dataset.tail(12)) #To print the last 12 rows

    print(dataset.dtypes) #To print the data types for each column

    print(dataset.columns[0]) #To print the first column name

    print(dataset.info()) #To print the data types for each column

    num_distinct_values = dataset['Age'].nunique() #Choose a categorical attribute and display the distinct values it contains
    distinct_values = dataset['Age'].unique()
    print("\nNumber of distinct values in 'Age': {}".format(num_distinct_values))
    print("Distinct values in 'Age':")
    print(distinct_values)

    print(dataset['Gender'].mode()) #To print the most frequently occurring value in the chosen categorical attribute

    column = 'Age' #To calculate the mean, median, standard deviation and the quantile for the select column
    print(dataset[column].mean())
    print(dataset[column].median())
    print(dataset[column].std())
    print(dataset[column].quantile(0.20))

    filtered_data = dataset[dataset['Age'] > 30] #Filter based on age
    print("Filtered data (Age > 30):")
    print(filtered_data)

    filtered_on_name = dataset[dataset['Gender'].str.startswith('F')] #To filter based on letter
    print(filtered_on_name)

    no_duplicates = dataset.drop_duplicates() #To remove the dupilcated rows
    print (no_duplicates)

    dataset['Age'] = dataset['Age'].astype(str) #To change the data type from int to str
    print(dataset.dtypes)

    grouped_data = dataset.groupby(['Gender', 'Marital_Status']).size().reset_index(name='Count') #Group data based on two attributes
    print(grouped_data)

    missing_values = dataset.isnull().sum() #Check for missing vals
    print("Missing values in the dataset:")
    print(missing_values)

    dataset['Age'].fillna(dataset['Age'].median(), inplace=True) #Replace missing values
    dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
    print("\nMissing values replaced.")

    missing_values = dataset.isnull().sum() #Check for missing vals
    print("Missing values in the dataset:")
    print(missing_values)

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median()) #Replace missing values
    dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode()[0])
    print("\nMissing values replaced.")

    missing_values = dataset.isnull().sum()  # Re-Check for missing vals
    print("Missing values in the dataset:")
    print(missing_values)

    dataset['Monthly_Income_bins'] = pd.cut(dataset['Monthly_Income'], bins=5) #Divide in bins and count
    bin_counts = dataset['Monthly_Income_bins'].value_counts().sort_index()
    print(bin_counts)

    max_row = dataset.loc[dataset['Monthly_Income'].idxmax()] #To find the row with the maximum value
    print(max_row)

    sns.boxplot(y=dataset['Monthly_Income']) #boxplot for Monthly Income since it's significant for Employee's satisfaction
    plt.title('Boxplot of Monthly Income')
    plt.xlabel('Monthly Income')
    plt.savefig('boxplot.png')

    sns.histplot(x = dataset['Monthly_Income']) #histogram for Monthly Income
    plt.title('Histogram of Monthly Income')
    plt.xlabel('Monthly Income')
    plt.savefig('histogram.png')

    plt.scatter(x = dataset['Monthly_Income'], y = dataset['Job_Satisfaction']) #scatter plot for Monthly Income and Job Satisfaction
    plt.title('Scatter Plot of Monthly Income and Job Satisfaction')
    plt.xlabel('Monthly Income scatter')
    plt.savefig('scatter.png')

    plt.figure(figsize=(8, 6))
    correlation_matrix = dataset[['Age', 'Monthly_Income', 'Job_Satisfaction']].corr()  # Compute correlation for all numerical columns using the heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

    numerical_features = dataset.select_dtypes(include=['number']) #Normalize numerical attributes
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features.columns)
    print(scaled_numerical_df.head().to_markdown(index=False, numalign="left", stralign="left"))
    print(scaled_numerical_df.info())


        # Select numerical features for PCA
    numerical_features = dataset.select_dtypes(include=['number'])
        # Impute missing values with the median of each column
    numerical_features = numerical_features.fillna(numerical_features.median())
        # Standardize the numerical features
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
        # Apply PCA with 2 components
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_numerical)
        # Create a DataFrame for the principal components
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
        # Display the first 5 rows
    print(principalDf.head().to_markdown(index=False, numalign="left", stralign="left"))
        # Print the column names and their data types
    print(principalDf.info())
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


    numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns #To calculate the correlation matrix
    correlation_matrix = dataset[['Monthly_Income', 'Job_Satisfaction']].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)


    dataset = pd.read_csv('employee_attrition_dataset.csv') #Dataset is balanced!
    gender_counts = dataset['Gender'].value_counts()
    print("Number of Males and Females:")
    print(gender_counts)


    dataset['Income_Satisfaction_Interaction'] = dataset['Monthly_Income'] * dataset['Job_Satisfaction']# Create new features
    dataset['Age_Squared'] = dataset['Age'] ** 2
    dataset['Income_Bin'] = pd.cut(dataset['Monthly_Income'], bins=[0, 6000, 8000, 10000], labels=['Low', 'Medium', 'High'])
    dataset['Income_to_Age_Ratio'] = dataset['Monthly_Income'] / dataset['Age']
    print("Dataset with New Features:")
    print(dataset)