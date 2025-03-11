import numpy
from pandas import read_csv

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

