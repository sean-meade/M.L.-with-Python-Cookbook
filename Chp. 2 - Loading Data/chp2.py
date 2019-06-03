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