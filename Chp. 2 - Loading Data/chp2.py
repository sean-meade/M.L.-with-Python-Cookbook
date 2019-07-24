# Loading a Sample Dataset
# Load scikit-learn's datasets
from sklearn import datasets

# Load digits dataset
digits = datasets.load_digits()

# Create features matrix
features = digits.data

# Create target vector 
target = digits.target

# View first observation
print(features[0])
#==> [ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
#     15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
#      0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
#      0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]

"""
Scikit-learn has "toy" datasets that can be used to test machine learning algorithms.
(http://bit.ly/2HS6Dzq , http://bit.ly/2mNSEBZ)
For example:
1. load_boston
    Contains 503 observations on Boston housing prices. It is a good dataset for 
    exploring regression algorithms.
2. load_iris
    Contains 150 observations on the measurements of Iris Flowers. It is a good data-
    set for exploring classification algorithms.
3. load_digits
    Contains 1,797 observations from images of handwritten digits. It is a good data-
    set for teaching image classification.
"""
#################################################################

# Creating a Simulated Dataset

"""
Scikit-learn has a different ways to create simulated data. Here are three useful methods.

When we want a dataset designed to be used with linear regression, make_regression is a 
good choice.
"""
# Load library
from sklearn.datasets import make_regression

# Generate features matrix, target vector, and the true coefficients
features, targer, coefficients = make_regression(n_samples = 100,
                                                    n_features = 3,
                                                    n_informative = 3,
                                                    n_targets = 1,
                                                    noise = 0.0,
                                                    coef = True,
                                                    random_state = 1)

# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# We can create a simulated dataset for classification, we can use make_classification

# Load Library
from sklearn.datasets import make_classification

# Generate features matrix and target vector
features, target = make_classification(n_samples = 100,
                                        n_features = 3,
                                        n_informative = 3,
                                        n_redundant = 0,
                                        n_classes = 2,
                                        weights = [.25, .75],
                                        random_state = 1)

# View feature matrix and target matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# Creating a dataset to work well with clustering techniques

# load library
from sklearn.datasets import make_blobs

# Generate feature matrix and target vector
features, target = make_blobs(n_samples = 100,
                                n_features = 2,
                                centers = 3,
                                cluster_std = 0.5,
                                shuffle = True,
                                random_state = 1)

# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# To view these clusters
# Load library
import matplotlib.pyplot as plt

# View scatterplot
# [:,0] means selecting all of the first column
plt.scatter(features[:,0], features[:,1], c=target)
plt.show()

# 2.3 Loading a CSV file
# Use pandas library's read_csv

# load library
import pandas as pd

# Create URL
url = "https://tinyurl.com/simulated_data"

# Load dataset
dataframe = pd.read_csv(url)

# View first two rows
dataframe.head(2)

# Notes: sep parameter for read_csv identifies what seperates the data in the csv file
#        header parameter for read_csv identifies where the header row is (if there is none then header=None)

# 2.4 Loading an Excel File
# Use pandas library's read_excel

# Load library
import pandas as pd

# Create URL
url = "https://tinyurl.com/simulated_excel"

# Load data
dataframe = pd.read_excel(url, sheetname = 0, header = 1)

# View the first two rows
dataframe.head(2)

# Notes: sheetname specifies which sheet is used
#        If more then one sheet is needed use as list (e.g. sheetname = [0,1,2, "Monthly Sales"])

# 2.5 Loading a JSON File
# Use pandas library's read_json

# Load library
import pandas as pd

# Create URL
url = "https://tinyurl.com/simulated_json"

# Load data
dataframe = pd.read_json(url, orient = 'columns')

# View the first two rows
dataframe.head(2)

# Notes: orient indicates how the JSON file is structured
#        json_normalize can help convert semistructured JSON data into a pandas DataFrame

# 2.6 Querying a SQL Database
# Use pandas library's read_sql_query

# Load libraries
import pandas as pd
from sqlalchemy import create_engine

# Create a connection to the database
database_connection = create_engine("sqlite:///sample.db")

# Load data
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)

# View first two rows
dataframe.head(2)

# Notes: create_engine defines a connection to a SQL database engine called SQLite
