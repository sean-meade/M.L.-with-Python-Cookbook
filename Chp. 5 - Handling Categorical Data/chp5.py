# Chapter 5 - Handling Categorical Data

# 5.0 Introduction
# It is often useful to measure objects not in terms of quantity but in terms of some quality. We frequently 
# represent this qualitative information as an observation's membership in a discrete category such as gender, 
# colors, or brand of car. However, not all categorical data is the same. Sets of categories with no intrinsic
# ordering is called nominal. Examples of nominal categories include:
#           - Blue, Red, Green
#           - Man, Woman
#           - Banana, Strawberry, Apple

# In contrast, when a set of categories has some natural ordering we refer to it as ordinal. For example:
#           - Low, Medium High
#           - Young, Old
#           - Agree, Neutral, Disagree

# Furthermore, categorical information is often represented in data as a vector or column of strings (e.g., "Maine",
# "Texas", "Delaware"). The problem is that most machine learning algorithms require inputs be numerical values.

# The k-nearest neighbor algorithm provides a simple example. One step in the algorithm is calculating the distance 
# between observations - often using the Euclidean distance. However, this distance can't be calculated with a 
# string. 

# 5.1 Encoding Nominal Categorical Features
# When you have a feature with nominal classes that has no intrinsic ordering (e.g., apple, pear, banana).
# One- hot encode the feature using scikit-learn's 'LabelBinarizer':
# Import libraries
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Create feature
feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])

# Create one-hot encoder
one_hot = LabelBinarizer()

# One-hot encode feature
one_hot.fit_transform(feature)

# We can use the 'classes_' method to output the classes:
# View feature classes
one_hot.classes_

# If we want to reverse the one-hot encoding, we can use inverse_transform:
# Reverse one-hot encoding
one_hot.inverse_transform(one_hot.transform(feature))

# We can even use pandas to one-hot encode the feature:
# Import libraries
import pandas as pd

# Create dummy variables from feature
pd.get_dummies(feature[:,0])

# One helpful ability of of scikit-learn is to handle a situation where each observation lists multiple classes:
# Create multiclass feature
multiclass_feature = np.array([("Texas", "Florida"),
                                ("California", "Alabama"),
                                ("Texas", "Florida"),
                                ("Delaware", "Florida"),
                                ("Texas", "Alabama")])

# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()

# One-hot encode multiclass feature
one_hot_multiclass.fit_transform(multiclass_feature)

# Once again, we can see the classes with the 'classes_' method:
# View classes
one_hot_multiclass.classes_

# You might think the proper strategy is to assign each class a numerical value (e.g., Texas = 1, California = 2).
# However, when when our classes have no intrinsic ordering (e.g., Texas isn't less than California), the numerical
# values erroneously create an ordering that is not present.

# The proper strategy is to create a binary feature for each classin the original feature. This is often called 
# one-shot encoding (in machine learning literature) or dummying (in statistical and research literature). Our 
# solution's feature was a vector containing three classes (i.e., Texas, California, and Delaware). In one-hot 
# encoding, each class becomes its own feature with 1s when the class appears and 0s otherwise. Because our feature
# had three classes, one-hot encoding returned three binary features (one for each class). By using one-hot encoding 
# we can capture the membership of an observation in a classs while preserving the notion that the class lacks any
# sort of hierarchy.

# Finally, it is worthwhile to note that it is often recommended that one-hot encoding a feature, we drop one of 
# the one-shot encoded features in the resulting matrix to avoid linear dependence. 
# See also : (http://bit.ly/2FvVJkC) & (http://bit.ly/2FwrxG0)

# 5.2 Encoding Ordinal Categorical Features
# When you have an ordinal categorical feature (e.g., high, medium, low).
# Use pandas DataFrame's 'replace' method to transform string labels to numerical equivalents:
# Load library
import pandas as pd

# Create features
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})

# Create mapper
scale_mapper = {"Low": 1,
                "Medium": 2,
                "High": 3}

# Replace feature values with scale
dataframe["Score"].replace(scale_mapper)

# Often we have a feature with classes that have some kind of natural ordering. A famous example is the Likert scale:
#       - Strongly Agree
#       - Agree
#       - Neutral
#       - Disagree
#       - Strongly Disagree

# When encoding the feature for use in machine learning, we need to transform the ordinal classes into numerical 
# values that maintain the notion of ordering. The most common approach is to create a dictionary that maps the 
# string label of the class to a number and then apply that map to the feature.

# It is important that our choice of numeric values is based on our prior information on the ordinal classes. In our
# solution, 'high' is literally three times larger than 'low'. This is fine in any instance, but can break down if 
# the assumed intervals between the classes are not equal:

dataframe = pd.DataFrame({"Score": ["Low",
                                    "Low",
                                    "Medium",
                                    "Medium",
                                    "High",
                                    "Barely More Than Medium"]})

scale_mapper = {"Low": 1,
                "Medium": 2,
                "Barely More Than Medium": 3,
                "High": 4}

dataframe["Score"].replace(scale_mapper)

# In this example, the distance between 'Low' and 'Medium' is the same as the distance between 'Medium' and 'Barely
# More Than Medium', which is almost certainly not accurate. The best approach is to be conscious about the numerical
# values mapped to classes:

scale_mapper = {"Low": 1,
                "Medium": 2,
                "Barely More Than Medium": 2.1,
                "High": 3}

dataframe["Score"].replace(scale_mapper)

# 5.3 Encoding Dictionaries of Features
# When you have a dictionary and want to convert it into a feature matrix
# Use 'DictVectorizer':
# Import library
from sklearn.feature_extraction import DictVectorizer

# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]

# Create dictionary vectorizer
dictvectorizer = DictVectorizer(sparse = False)

# Convert dictionary to feature matrix
features = dictvectorizer.fit_transform(data_dict)

# View feature matrix
features

# By default 'DictVectorizer' outputs a sparse matrix that only stores elements with a value other than 0. This
# can be very helpful when we have massive matrices (often encountered in natural language processing) and want 
# to minimize the memory requirements. We can force 'DictVectorizer' to output a dense matrix using 'sparse = False'.

# We can get the names of each generated feature using the get_feature_names method:
# Get feature names
feature_names = dictvectorizer.get_feature_names()

# View feature names
feature_names

# While not necessary, for the sake of illustration we can create a pandas DataFrame to view the output better:
# Import library
import pandas as pd

# Create dataframe from features
pd.DataFrame(features, columns = feature_names)

# A dictionary is popular data structure used by many programming languages; however, machine learning algorithms
# expect the data to be in the form of a matrix. We can accomplish this using scikit-learn's 'dictvectorizer'.

# This is a common situation when working with natural language processing. For example, we might have a collection 
# of documents and for each document we have a dictionary containing the numberof times every word appears in the 
# document. Using dictvectorizer, we can easily create a feature matrix where every feature is the number of times a 
# word appears in each document:

# Create word counts dictionaries for four documents
doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}

# Create list of word count dictionaries into feature matrix
doc_word_counts = [doc_1_word_count,
                    doc_2_word_count,
                    doc_3_word_count,
                    doc_4_word_count]

# Convert list of word count dictionaries into feature matrix
dictvectorizer.fit_transform(doc_word_counts)#

# In our toy example there are only three unique words (Red, Yellow, Blue) so there are only three features in our 
# matrix; however, you can imagine that if each document was actually a book in a university library our feature 
# matrix would be very large (and then we would want to set spare to True).

# See Also: (http://bit.ly/2HReoWz) & (http://bit.ly/2HReBZR)

# 5.4 Imputing Missing Class Values
# When you have a categorical feature containing missing values that you want to replace with predicted values.
# The ideal solution is to train a machine learning classifier algorithm to predict the missing values, commonly
# a k-nearest neighbors (KNN) classifier:
# Load libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])

# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])

# Train KNN learner
clf = KNeighborsClassifier(3, weights = 'distance')
trained_model = clf.fit(X[:,1:], X[:,0])

# Predict missing values' class
imputed_values = trained_model.predict(X_with_nan[:,1:])

# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))

# Join two features matrices
np.vstack((X_with_imputed, X))

# An alternative solution is to fill in missing values with the feature's most frequent value:
from sklearn.preprocessing import Imputer

# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))

imputer = Imputer(strategy = 'most_frequent', axis = 0)

imputer.fit_transform(X_complete)

# When we have missing values in a categorical feature, our best solution is to open our toolbox of machine learning
# algorithms to predict the values of the missing observations. We can accomplish this by treating the feature with 
# the missing values as the target vector and the other features as the feature matrix. A commonly used algorithm is
# KNN, which assigns to the missing value the median class of the k nearest observations.

# Alternatively, we can fill in missing values with the most frequent class of the feature. While less sophisticated
# than KNN, it is much more scalable to larger data. In either case, it is advisable to include a binary feature 
# indicating which observations contain imputed values.

# See Also: (http://bit.ly/2HSsNBF) & (http://bit.ly/2HS9sAT)

# 5.5 Handling Imbalanced Classes
# When you have a target vector with highly imbalanced classes.
# Collect more data. If that isn't possible, change the metrics used to evaluate your model. If that doesn't work, 
# consider using a model's built-in class weight parameters (if available), downsampling, or upsampling. We cover 
# evaluation metrics in a later chapter, so far now let us focus on class weight parameters, downsampling and 
# upsampling.

# To demonstrate our solutions, we need to create some data with imbalanced classes. Fisher's Iris dataset conatains
# three balanced classes of 50 observations, each indicating the species of flower (Iris setosa, Iris virginica, and 
# Iris verssicolor). To unbalance the dataset, we remove 40 of the 50 Iris setosa observations and then merge the Iris
# virginica and Iris versicolor classes. The end result is 10 observations of Iris setosa (class 0) and 100 
# observations of not Iris setosa (class 1):
# Load libraries 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load Iris data
iris = load_iris()

# Create feature matris
features = iris.data

# Create target vector
target = iris.target

# Remove first 40 observations
features = features[40:,:]
target = target[40:]

# Create binary target vector indicating if class 0
target = np.where((target == 0), 0, 1)

# Look at the imbalanced target vector
target

# Many algorithms in scikit-learn offer a parameter to weight classes during training to counteract the effect of 
# their imbalance. While we have not covered it yet, RandomForestClassifier is a popular classification algorithm
# and includes a class_weight parameter. You can pass an argument specifying the desired class weights explicity:
# Create weights
weights = {0: .9, 1: 0.1}

# Create random forest classifier with weights
RandomForestClassifier(class_weight = weights)

RandomForestClassifier(bootstrap = True, class_weight = {0: .9, 1: 0.1},
                        criterion = 'gini', max_depth = None, max_features = 'auto',
                        max_leaf_nodes = None, min_impurity_decrease = 0.0,
                        min_impurity_split = None, min_samples_leaf = 1,
                        min_samples_split = 2, min_weight_fraction_leaf = 0.0,
                        n_estimators = 10, n_jobs = 1, oob_score = False, random_state = None,
                        verbose = 0, warm_start = False)

# Or you cab pass balanced, which automatically creates weights inversely proportional to class frequencies:
# Train a random forest with balanced class weights
RandomForestClassifier(class_weight = "balanced")

RandomForestClassifier(bootstrap = True, class_weight = 'balanced',
                        criterion = 'gini', max_depth = None, max_features = 'auto',
                        max_leaf_nodes = None, min_impurity_decrease = 0.0,
                        min_impurity_split = None, min_samples_leaf = 1,
                        min_samples_split = 2, min_weight_fraction_leaf = 0.0,
                        n_estimators = 10, n_jobs = 1, oob_score = False, random_state = None,
                        verbose = 0, warm_start = False)

# Alternatively, we can downsample the majority class or upsample the minority class. In downsampling, we randomly 
# sample without replacement from the majority class (i.e., the class with more observations) to create a new subset 
# of observations equal in size to the minority class. For example, if the minority class has 10 observations, we
# will randomly select 10 observations from the majority class and use those 20 observations as our data. Here we do
# Here we do exactly that using our unbalanced Iris data:
# Indicies of each class' observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

# NUmber of observations in each class
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# For every observation of class 0, randomly sample from class 1 without replacement
i_class1_downsampled = np.random.choice(i_class1, size = n_class0, replace = False)

# Join together class 0's target vector with the downsampled class 1's target vector
np.hstack((target[i_class0], target[i_class1_downsampled]))

# Join together class 0's feature matrix with the downsampled class 1's target matrix
np.vstack((features[i_class0,:], features[i_class1_downsampled]))[0:5]

# Our other option is to upsample the minority class. In upsampling, for every observation in the majority class, 
# we randomly select an observation from the minority class with replacement. The end result is the same number
# of observations from the minority and majority classes. Upsampling is implemented very similarly to downsampling,
# just in reverse.
# For every observation in class 1, randomly sample from class 0 with replacement
i_class0_upsampled = np.random.choice(i_class0, size = n_class1, replace = True)

# Join together class 0's upsampled target vector with class 1's target vector
np.concatenate((target[i_class0_upsampled], target[i_class1]))

# Join together class 0's upsampled feature matrix with class 1's feature matrix
np.vstack((features[i_class0_upsampled, :], features[i_class1, :]))[0:5]

# In the real world, imbalanced classes are everywhere - most visitors don't click the buy button and many types of
# cancer are thankfully rare. For this reason, handling imbalanced classes is a common activity in machine leanring.

# Our best strategy is simply to collect more observations - especially observations from the minority class. However,
# this is often just not possible, so we have to resort to other options. 

# A second strategy is to use a model evaluation metric better suited to imbalanced classes. Accuracy is often used 
# as a metric for evaluating the performance of a model, but when imbalanced classes are present accuracy can be ill 
# suited. For example, if only 0.5% of observations have rare cancer, then even a naive model that predicts nobody 
# has cancer will be 99.5% accurate. Clearly this is not ideal. Some better metrics we discuss in later chapters are
# confusion matrices, precision, recall, F1 scores, and ROC curves.

# A third strategy is to use the class weighing parameters included in implementationsof some models. This allows us 
# to have algorithm adjust fro imbalanced classes. Fortunately, many scikit-learn classifiers have a 'class_weight' 
# parameter, making it a good option.

# The fourth and fifth strategies are related: downsampling and upsampling. In downsampling we create a random subset
# of the majority class of equal size to the minority class. In upsampling we repeatedly sample with replacement from
# the minority class to make it of equal size as the majority class. The decision between using downsampling and 
# upsampling is context-specific, and in general we should try both to see which produces better results.