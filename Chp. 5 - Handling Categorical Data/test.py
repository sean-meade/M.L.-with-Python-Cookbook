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
print(np.vstack((features[i_class0_upsampled, :], features[i_class1, :]))[0:5])

# In the real world, imbalanced classes are everywhere - most visitors don't click the buy button and many types of
# cancer are thankfully rare. For this reason, handling imbalanced classes is a common activity in machine leanring.

# Our best strategy is simply to collect more observations - especially observations from the minority class. However,
# this is often just not possible, so we have to resort to other options. 