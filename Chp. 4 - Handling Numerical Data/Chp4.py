# Chapter 4 - Handling Numerical Data
# Quantitive data is the measurement of something - whether class size, monthly sales, or student scores. The 
# natural wat to represent these quantities is numerically (e.g., 29 students, $529,392 in sales). In this chapter,
# we will cover numerous strategies for transforming raw numerical data into features purpose-built for machine
# learning algorithms.

# 4.1 Rescaling a Feature (http://bit.ly/2FwwRcM)
# Use scikit-learn's 'MinMaxScalar' to rescale a feature between two values

# Load libraries
import numpy as np
from sklearn import preprocessing

# Create feature
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# Create Scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range = (0, 1))

# Scale feature
scaled_feature = minmax_scale.fit_transform(feature)

# Show feature
scaled_feature

# Many algorithms use scaled features.
# You can also use 'fit' to calculate max and min values of a feature, then use transform to rescale the feature.
# This method allows you to apply the same transformation to different sets of the data.

# 4.2 Standardizing a Feature
# You want to transform a feature to have a mean of 0 and a standard deciation of 1.
# scikit-learn's 'StandardScaler' performs both transformations.

# Load libraries
import numpy as np
from sklearn import preprocessing

# Create feature
x = np.array([[-1000.1],
                [-200.2],
                [500.5],
                [600.6],
                [9000.9]])

# Create Scaler
scaler = preprocessing.StandardScaler()

# Transform the feature
standardized = scaler.fit_transform(x)

# Show feature
standardized

# Print mean and standard deviation
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())

# If your data has significant outliers, it can negatively impct the standardization by affecting the
# feature's mean and variance. In this scenario, it is often helpful to instead rescale the feature using the median
# and quartile range. In scikit-learn we do this using the RobustScaler method:

# Create scaler
robust_scaler = preprocessing.RobustScaler()

# Transform feature
robust_scaler.fit_transform(x)

# 4.3 Normalizing Observations
# When you want to rescale the feature values of observations to have unit norm (a total length of 1)
# Use 'Normalizer' with a 'norm' argument

# Load libraries
import numpy as np
from sklearn.preprocessing import Normalizer

# Create feature matrix
features = np.array([[0.5, 0.5],
                    [1.1, 3.4],
                    [1.5, 20.2],
                    [1.63, 34.4],
                    [10.9, 3.3]])

# Create notmalizer
normalizer = Normalizer(norm = "l2")

# Transfor feature matrix
normalizer.transform(features)

# Many rescaling methods (e.g., min-max scaling and standardization) operate on features; however, we can also 
# rescale across individual observations. 'Normalizer' rescales the values on individual observations to have 
# unit norm (the sum of their lengths is 1). This type of rescaling is often used when we have many equivalent
# features (e.g., text classification when every word of n-word group is a feature).

# 'Normalizer' provides three 'norm' options with Euclidean norm (often called L2) being the default argument. 
# Alternatively, we can specify Manhattan norm (L1).

# Transform feature matrix
features_l2_norm = Normalizer(norm = "l1").transform(features)

features_l1_norm = Normalizer(norm = "l1").transform(features)

# l2 = as the bird flies
# l1 = walking the streets

# Partically, notice that 'norm = "l1"' rescales an observation's values so theysum to 1, which can sometimes be
# a desirable quality:

# Print sum
print("Sum of the first observation\'s values:", features_l1_norm[0, 0] + features_l1_norm[0, 1])

# 4.4 Generating Polynomial and Interaction Features
# When you want to create polynominal and interaction features.
# Even though some choose to create polynominal and interaction features manually, scikit-learn 
# offers a built-in method

# Load libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create feature matrix
features = np.array([[2, 3],
                    [2, 3],
                    [2, 3]])

# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree = 2, include_bias = False)

# Create polynomial features
polynomial_interaction.fit_transform(features)

# 'degree' determines the degree of the polynomial (i.e., the raised power)
# By default 'PolynomialFeatures' includes interaction features.
# You can restrict the features created to only interaction features by setting 'interaction_only = True':
interaction = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
interaction.fit_transform(features)

# Polynomial features are often created when we want to include the notion that there exists a nonlinear 
# relationship between the features and the target. For example, we might suspect that the effect of age 
# on the probability of having a major medical condition is not constant over time but increases as age increases.
# We can encode that nonconstant effect in a feature, x, by generating that feature's higher-order forms (x squared,
# x cubed, etc.).

# Additionally, often we run into situations where the effect of one feature is dependenton another feature. A simple 
# example would be if we were trying to predict whether or not our coffee was sweet and we had two features: 
# 1) whether or not the coffee was stirred and 2) if we added sugar. Individualy, each feature does not predict coffee
# sweetness, but the combination of their effects does. That is, a coffee would only be sweet if the coffee had sugar
# and was stirred. The effects of each feature on the target (sweetness) are dependent on each other. We can encode
# that relationship by including an interaction feature that is the product of the individual features.
"""#####################################################"""
# 4.5 Transforming Features
# When you want to make a custom transformation to one or more features.
# In scikit-leanr, use 'FunctionTransformer' to apply a function to a set of features.

# Load libraries
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Create feature matrix
features = np.array([[2, 3],
                    [2, 3],
                    [2, 3]])

# Define a simple function
def add_ten(x):
    return x + 10

# Create transformer
ten_transformer = FunctionTransformer(add_ten)

# Transform feature matrix
ten_transformer.transform(features)

# We can create the same transformation in pandas using 'apply':
# Load library
import pandas as pd

# Create DataFrame
df = pd.DataFrame(features, columns = ["feature_1", "feature_2"])

# Apply function
df.apply(add_ten)

# It is common to want to make some custom transformations to one or more features. For example, we might want to 
# create a feature that is the natural log of the values of the different feature. We can do this by creating a 
# function and then mapping it to features using either scikit-learn's 'FunctionTransformer' or pandas' 'apply'.
# In the solution we created a very simple function, add_ten, which added 10 to each input, but there is no reason
# we could not define a much more complex function.

# 4.6 Detecting Outliers
# When you want to identify extreme observations
# Detecting outliers is unfortunately more of an ary than a science. However a common method is to assume the data 
# is normally distributed and based on that assumption "draw" an allipse around the data, classifying any observation
# inside the ellipse as an inlier (labeled as 1) and any observation outside the ellipse as an outlier (labeled as -1).

# Load libraries
import numpy as np 
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# Create simulated data
features, _ = make_blobs(n_samples = 10,
                            n_features = 2,
                            centers = 1,
                            random_state = 1)

# Replace the first obserrvation's values with the extreme values
features[0,0] = 10000
features[0,1] = 10000

# Create detector
outlier_detector = EllipticEnvelope(contamination = .1)

# Fit detector
outlier_detector.fit(features)

# Predict outliers
outlier_detector.predict(features)

# A major limitation to this approach is the need to specify a 'contamination' parameter, which is the proportion
# observations that are outliers - a value that we don't know. Think of 'contamination' as our estimate of the 
# cleaniness of our data. If we expect out data to have few outliers, we can set 'contamination' to something small.
# However, if we believe that the data is very likely to have outliers, we can set it to a higher value.

# Instead of looking at observations as a whole, we can instead look at individual features and identify extreme
# values in those features using interquartile range (IQR):

# Create one feature
feature = features[:,0]

# Create a function to return index of outliers
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

# Run Function
indicies_of_outliers(features)

# IQR is the difference between the first and the third quartile of a set of data. You can think of IQR as the 
# spread of the bulk of the data, with outliers being observationsfar from the main concentration of data. 
# Outliers are commonly defined as any value 1.5 IQRs less than the first quartile or 1.5 IQRs greater than the 
# third quartile.

# There is no single best technique for detecting outliers. Instead, we have a collection of techniques all with 
# their own advantages and disadvantages. Our best strategy is often trying multiple techniques (e.g., both 
# 'EllipticEnvelope' and IQR-based detection) and looking at the results as a whole.

# If at all possible, we should take a look at observations we detect as outliers and try to understand them. For
# example, if we have a dataset of houses and one feature is numberof rooms, is an outlier with 100 rooms really a 
# house or is it actually a hotel that has been misclassified? (http://bit.ly/2FzMC2k).

# 4.8 Discretizating Features
# When you have a numerical feature and want to break it up into discrete bins.
# Depending on how we want to break up the data, there are two techniques we can use. First, we can binarize 
# the feature according to some threshold:

# Load libraries
import numpy as np
from sklearn.preprocessing import Binarizer

# Create feature
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

# Create binarizer
binarizer = Binarizer(18)

# Transform feature
binarizer.fit_transform(age)

# Second, we can break up numerical features according to multiple thresholds:
# Bin feature
np.digitize(age, bins = [20, 30, 64])

# Note that the arguments for the bins parameter denote the left edge of each bin. For example, the 20 argument
# does not include the element with the value 20, only the two values smaller than 20. We can switch this behavior
# by setting the parameter right to True:
# Bin feature
np.digitize(age, bins = [20, 30, 64], right = True)

# Discretization can be a fruitful strategy when we have reason to believe that a numerical feature should behave 
# more like a categorical feature. For example, we might believe there is very little difference between 20 and 
# 21-year-olds. In that example, it could be useful to break up individuals in our data into those who can drink 
# alcohol and those who cannot. Similarly, in other cases it might be useful to discretize our data into three or 
# more bins.

# In the solution, we saw two methods of discretization - scikit-learn's 'Binarizer' for two bins and NumPy's 
# 'digitize' (http://bit.ly/2HSciFP) for three or more bins - however, we can also use 'dititize' to binarize features like 'Binarizer' by
# only specifying a single threshold:
# Bin feature
np.digitize(age, bins = [18])

# 4.9 Grouping Observations Using Clustering
# When you want to cluster observations so that similar observations are grouped together.
# If you know that you have k groups, you can use k-means clustering to group similar observations and output a 
# new feature containing each observation's group membership:

# Load libraries
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Make simulated feature matrix
features, _ = make_blobs(n_samples = 50,
                            n_features = 2,
                            centers = 3,
                            random_state = 1)

# Create DataFrame
dataframe = pd.DataFrame(features, columns = ["feature_1", "feature_2"])

# Make k-means clusterer
clusterer = KMeans(3, random_state = 0)

# Fit clusterer
clusterer.fit(features)

# Predict values
dataframe["group"] = clusterer.predict(features)

# View first few observations
dataframe.head(5)

# Clustering algorithms are explained more in chapter 19. We can use clustering as a preprocessing step.
# Specifically, we use unsupervised learning algorithms like k-means to cluster observations into groups. The end
# result is a categorical feature with similar observations being members of the same group.

# 4.10 Deleting observations with Missing Values
# When you need to delete observations containing missing values.
# Deleting observations with missing values is easy with a clever line of NumPy:

# Load library
import numpy as np

# Create feature matrix
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])

# Keep only observations that are not (denoted by ~) missing
features[~np.isnan(features).any(axis = 1)]

# Alternatively, we can drop missing observations using pandas:
# Load library
import pandas as pd

# Load data
dataframe = pd.DataFrame(features, columns = ["feature_1", "feature_2"])

# Remove observations with missing values
dataframe.dropna()

# Most machine learning algorithms cannot handle any missing values in the target and feature arrays. 
# Deleting observations can in some cases introduce bias into our data. There are three types of missing data:
# (http://bit.ly/2Fto4bx) & (http://bit.ly/2FAkKlI)

# Missing Completely At Random (MCAR): The probability that a value is missing is independent of everything.

# Missing At Random (MAR): The probability that a value is missing is not completely random, but depends on 
#                          the information captured in other features.

# Missing Not At Random (MNAR): The probability that a value is missing is not random and depends on information
#                               not captured in our features.

# It is sometimes acceptable to delete observations if they are MCAR or MAR. However, if the value is MNAR, the 
# fact that a value is missing is itself information. Deleting MNAR observations can inject bias into our data 
# becasue we are removing observations produced by some unobserved systematic effect.

# 4.11 Imputing Missing Values
# When you have missing values in your data and want to fill in or predict their values.
# If you have a small amount of data, predict the missing values using k- nearest neighbors (KNN):
# Load libraries
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Make a simulated feature matrix
features, _ = make_blobs(n_samples = 1000,
                            n_features = 2,
                            random_state = 1)

# Stanardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Replace the first feature's first value with a missing value
true_value = standardized_features[0, 0]
standardized_features[0, 0] = np.nan

# Predict the missing values in the feature matrix
features_knn_imputed = KNN(k = 5, verbose = 0).fit_transform(standardized_features)

# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value", features_knn_imputed[0, 0])

# Alternatively, we can use scikit-learn's Imputer module to fill in missing values with the features mean, 
# median, or most frequent value. However, we will typically get worse results than KNN:
# Load libraries
from sklearn.preprocessing import Imputer

# Create imputer
mean_imputer = Imputer(strategy = "mean", axis = 0)

# Impute values
features_mean_imputed = mean_imputer.fit_transform(features)

# Compare true to imputed values
print("True Value:", true_value)
print("Imputed value:", features_mean_imputed[0, 0])

# There are two main strtegies for replacing missing data with substitute values, each of which has strengths 
# and weaknesses. First, we can use machine learning to predict the values of the missing data. To do this we treat
# the feature with missing values as a target vector and use the remaining subset of features to predict misisng 
# values. While we can use a wide range of machine learning algorithms to impute values, a popular choice is KNN.
# KNN is addressed in chapter 14. In short the algorithm uses k nearest observations (according to some distance
# metric) to predict the missing value. In our solution we predict the missing value using the five closest 
# observations.

# The downside to KNN is that in order to know which observations are the closest to the missing value, it needs to
# calculate the distance between the missing value and every single observation. This is reasonable in smaller 
# datasets, but quickly becomes problematic if the dataset has millions of observations.

# An alternative and more scalable strategy is to fill in all missing values with some average value. For example,
# in our solution we used scikit-learn to fill in missing values with a feature's mean value. The imputed value is 
# often not as close to the true value as when we used KNN, but we can scale mean-filling to data containing 
# millions of observations easily.

# If we use imputation, it is a good idea to create a binary feature indicating whether or not the observation 
# contains an imputed value.

# A study of K-Nearest Neigbour as an Imputation Method (http://bit.ly/2HS9sAT)