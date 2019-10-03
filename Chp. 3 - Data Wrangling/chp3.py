# Data Wrangling
# Adapted from the book: Machine Learning with Python Cookbook by Chris Albon
# 3.0 Introduction
# Example of wrangled data in the form of a a data frame (tabular)

# Load library
import pandas as pd

# Create URL
url = "https://tinyurl.com/titanic-csv"

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show first 5 rows
dataframe.head(5)

# 3.1 Creating a Data Frame
# Creating an empty data frame using 'DataFrame' and defining each column after

# load library
import pandas as pd

# Create DataFrame
dataframe = pd.DataFrame()

# Add Columns
dataframe["Name"] = ["Jacky Jackson", "Steven Stevenson"]
dataframe["Age"] = [38, 25]
dataframe["Driver"] = [True, False]

# Show DataFrame
dataframe

'''Alternatively'''
# Append new rows to the created DataFrame object

# Create row 
new_person = pd.Series(['Molly Monney', 40, True], index = ['Name', 'Age', 'Driver'])

# Append row
dataframe.append(new_person, ignore_index = True)

# Show dataframe
dataframe

# 3.1 Describing the Data
# Viewing the first few rows using 'head', the number of rows and columns using 'shape' and
# statisics for any numeric column with 'describe'.

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Show first two rows
dataframe.head(2)

# Show the number of rows and columns (i.e. dimensions)
dataframe.shape

# Show statisics
dataframe.describe()

# Notes: 'tail' will show the last few rows

# 3.3 Navigating DataFrames
# Selecting individual data or slices of a DataFrame using 'loc' and 'iloc'

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Select first row
dataframe.iloc[0]

# Showing the second third and fourth row
dataframe.iloc[1:4]

# Show first four rows
print(dataframe.iloc[:4])

# You can set the index of a dataframe to anything where the value is unique to each row (like passenger name)
# Set index
dataframe = dataframe.set_index(dataframe['Name'])

# Show row
dataframe.loc['Allen, Miss Elisabeth Walton']

# Notes: 'loc' is useful when the index of the DataFrame is a label (e.g., string)
#        'iloc' works by looking for the position

# 3.4 Selecting Rows Based on Conditionals

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Show top two rows where column 'sex' is 'female'
dataframe[dataframe['Sex'] == 'female'].head(2)

# Notes: dataframe['Sex'] == 'female' is the condition.
#        (dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65) is how to set multiple conditions

# 3.5 Replacing Data
# Using pandas librarys 'replace' you can find and replace values

# We can replace any instance of "female" in the Sex column with "Woman".

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Replace values, show two rows
dataframe['Sex'].replace("female", "Woman").head(2)

# Replace multiple values
# Replacing "female" and "male" with "Woman" and "Man"
dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5)

# Can also find and replace across the entire DataFrame object by specifying the whole data frame instead of a single column
# Replace values, show two rows
dataframe.replace(1, "One").head(2)

# 'replace' also accepts regular expressions
# Replace values, show two rows
dataframe.replace(r"1st", "First", regex=True).head(2)

# 3.6 Renaming Columns
# Use 'rename' to rename column

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Rename column, show two rows
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)

# The rename method accepts a dictionary. Can use this to change multiple column names at once
dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)

# If you want to rename all the columns at once you can create a dictionary with the old column names
# as keys and empty strings as values

# Load library
import collections

# Create dictionary
column_names = collections.defaultdict(str)

# Create keys
for name in dataframe.columns:
    column_names[name]

# Show dictionary
column_names

# 3.7 Finding the Minimum, Maximum, Sum, Average, and Count
# Pandas comes with some built-in methods for commonly used descriptive statistics

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Calculate statistics
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min)
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())

# Notes: In addition pandas offers variance ('var'). standard deviation ('std'), kurtosis ('kurt'), skewness (skew),
# standard error of the mean ('sem'), mode ('mode'), median ('median') and a number of others.

# We can also apply these to the entire dataframe
# Show counts
dataframe.count()

# 3.8 Finding Unique Values
# Using the 'unique' method to view an array of all unique values in a column

# Load libray
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Select unique values
dataframe['Sex'].unique()

# Also 'value_counts' will display all unique values with the number of times each value shows

# Show counts
dataframe['Sex'].value_counts()

# Show number of unique with 'nunique'
dataframe['PClass'].nunique()

# 3.9 Handling Missing Values
# 'isnull' and 'notnull' return booleans indicating whether a value is missing:

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load Data
dataframe = pd.read_csv(url)

# Select missing values, show two rows ('notnull' will show rows with no missing values)
dataframe[dataframe['Age'].isnull()].head(2)

# To have full functionality over NaN you need to import NumPy library

# Load Library
import numpy as np

# Replace valueswith NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)

# Sometimes a data set uses a specific value to denote a missing observation.
# We can set these with a parameter in read_csv.

# Load data, set missing values
dataframe = pd.read_csv(url, na_values = [np.nan, 'NONE', -999])

# 3.10 Deleting a Column
# Using drop with parameter axis=1 (i.e., the column axis) you can delete a column

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Delete column
dataframe.drop('Age', axis = 1).head(2)

# Can also delete a list of columns
dataframe.drop(['Age', 'Sex'], axis = 1).head(2)

# If a column has no name you can drop it by its column index using dataframe.columns
dataframe.drop(dataframe.columns[1], axis = 1).head(2)

# Notes: Treat DataFrames as immutable objects and never use the arguement inplace=True

# 3.11 Deleting a Row
# Use a boolean condition to create a DataFrame excluding the rows you want to delete.

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Delete rows, show first two rows of output
dataframe[dataframe['Sex'] != 'male'].head(5)

# You can use 'drop' (for example df.drop([0, 1]) to drop the first two rows) but a more practical method is to
# is simply wrap a boolean inside df[]. 

# We can use conditionals to delete single rows by matching a unique value
# Delete row, show first two rows of output
dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2)

# Can use it to delete a row by row index
# Delete row, show first two rows of output
dataframe[dataframe.index != 0].head(2)

# 3.12 Dropping Duplicate rows
# Use 'drop_duplicates', but be minful of the parameters

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Drop duplicates, show first two rows
dataframe.drop_duplicates().head(2)

# Check to see numbers before and after
print("Number Of Rows In The Original Dataframe:", len(dataframe)) #==> 1313
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates())) #==> 1313
# This is because all rows are unique

# To check over a certain column
# Drop duplicates
dataframe.drop_duplicates(subset = ['Sex'])
# This will return the first female and male and discard the rest
dataframe.drop_duplicates(subset = ['Sex'], keep = 'last')
# This will return the last

# Return a boolean whether it's a duplicate or not
dataframe.duplicated()

# 3.13 Grouping Rows by Values
# 'groupby' is one of the most powerful features in pandas

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Group rows by the values of the column 'Sex', calculate mean of each group
dataframe.groupby('Sex').mean()

# Note: dataframe.groupby('Sex') produces ==> <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000024C1826F8D0>
#       This is because 'groupby' must be used with an operation we can use to each group.
#       Also using dataframe.groupby('Survived')['Name'].count() will group by survived or not and count names.
#       Can also group by one column and then group those groups by a second. For example 
#       dataframe.groupby(['Sex', 'Survived'])['Age'].mean() .

# 3.14 Grouping Rows by Time
# Use 'resample' to group individual rows by chunks of time

# Load libraries
import pandas as pd
import numpy as np

# Create data range: 30S means 30 second intervals, periods is the amount of of created entries and the date is staring date
time_index = pd.date_range('06/06/2017', periods = 100000, freq = '30S')

# Create DataFrame
dataframe = pd.DataFrame(index = time_index)

# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
dataframe.resample('W').sum()

# Notes: Resample requires index to be datatime-like values.
#        Can group by 2 weeks ('2W') or by month ('M')
#        Resample returns the last date of the group as index. To use first we use the 'label' parameter
#        dataframe.resample('M', label = 'left').count()

# 3.15 Looping Over a Column
# You can treat a pandas column like any other sequence in Python

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Print first two names uppercased
for name in dataframe['Name'][0:2]:
    print(name.upper())

# Also can create a list comprehension (a new list from an existing one)
# Show first two names in uppercase
[name.upper() for name in dataframe['Name'][0:2]]

# 3.16 Applying a Functiuon Over All Elements in a Column
# Use 'apply' to apply a built-in or custom function on every element in a column

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Create function
def uppercase(x):
    return x.upper()

# Apply function, show two rows
dataframe['Name'].apply(uppercase)[0:2]

# 3.17 Applying a Function to Groups
# Combine groupby and apply

# Load library
import pandas as pd

# Create URL
url = 'https://tinyurl.com/titanic-csv'

# Load data
dataframe = pd.read_csv(url)

# Group rows, apply function to groups
dataframe.groupby('Sex').apply(lambda x: x.count())

# 3.18 Concatenating DataFrames
# Use 'concat' with 'axis = 0' to concatenate along the row axis

# Load library
import pandas as pd

# Create Dataframe
data_a = {'id': ['1', '2', '3'],
            'first': ['Alex', 'Amy', 'Allen'],
            'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# Create Dataframe
data_b = {'id': ['4', '5', '6'],
            'first': ['Billy', 'Brian', 'Bran'],
            'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

# Concatenate DataFrames by rows
pd.concat([dataframe_a, dataframe_b], axis = 0)

# You can use 'axis = 1' to concatenate along the column axis
# Concatenate DataFrames by columns
pd.concat([dataframe_a, dataframe_b], axis = 1)

# We can use 'append' to add a new row to a DataFrame
# Create row
row = pd.Series([10, 'Chris', 'Chillon'], index = ['id', 'first', 'last'])

# Append row
dataframe_a.append(row, ignore_index = True)

# 3.19 Merging DataFrames (http://bit.ly/2Fuo4rH)
# To inner join, use 'merge' with the 'on' parameter to sspecify the column to merge on
# Inner join just combines the two with same employee_id and information in both

# Load library
import pandas as pd

# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
                    'name': ['Amy Jones', 'Allen Keys', 'Alice Bees', 'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id', 'name'])

# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
                'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id', 'total_sales'])

# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on = 'employee_id')

# 'merge' defaults to inner joins.
# To do an outer join use the 'how' parameter
# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on = 'employee_id', how = 'outer')

# For left and right joins
pd.merge(dataframe_employees, dataframe_sales, on = 'employee_id', how = 'left')

# Can also specify the column name in each DataFrame to merge on
pd.merge(dataframe_employees, dataframe_sales, left_on = 'employee_id', right_on = 'employee_id')

# Instead of merging on two columns we want to merge on the indexes of each DataFrame, we can replace 
# 'left_on' and 'right_on' parameters with 'right_index = True' and 'left_index = True'.
# If the two columns to be merged on have different names that's when 'left_on' and 'right_on' are used. Left
# is the first dataframe mentioned.

# Types of Joins:
# - Inner:
#       Return only rows that match in both DataFrames
# - Outer:
#       Return all rows in both DataFrames. Fill in NaN if no values exist
# - Left:
#       Return all rows in left but only rows from right DataFrame that matched with the left.
# - Right:
#       Return all rows in right but only rows from left DataFrame that matched with the right.