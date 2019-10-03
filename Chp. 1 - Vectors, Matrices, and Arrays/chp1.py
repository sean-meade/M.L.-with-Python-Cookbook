# The #==> symbol shows the output of the code beside it or above it
# Adapted from the book: Machine Learning with Python Cookbook by Chris Albon
# Creating a Vector & a Matrix

import numpy as np 

# Create a vector as a row
vector_row = np.array([1, 2, 3])

# Create a vector as a column
vector_column = np.array([[1], [2], [3]])

# Creating a Matrix
matrix = np.array([[1, 2],
                    [1, 2],
                    [1, 2]])
##################################################################
# Creating a sparse matrix
# Sparse matrices only store nonzero values

from scipy import sparse

zMatrix = np.array([[0, 0],
                    [0, 1],
                    [3, 0]])

# Create compressed sparse row (CPR) matrix
matrix_sparse = sparse.csr_matrix(zMatrix)

# View sparse matrix
print(matrix_sparse)
# Output: (location within matrix)   value 
# ==> (1, 1)   1
# ==> (2, 0)   3
##################################################################
# Selecting Elements
# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])

# Create matrix
matrix2 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

# Select third element of vector
vector[2] # ==> 3

# Select second row, second column
matrix2[1, 1] # ==> 5

# Select all elements in a vector
vector[:] # ==> array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Select everything up and including the third element
vector[3:] # ==> array([1, 2, 3])

# Select everything after the third element
vector[:3] # ==> array([4, 5, 6])

# Select the last element
vector[-1] # ==> 6

# select all rows and the second column
matrix2[:,1:2]
# ==> array([[2],
#           [5],
#           [6]])
##################################################################
# Describing a Matrix

# Create matrix
matrix3 = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# View number of rows and columns
matrix3.shape # ==> (3, 4)

# View number of elements (rows * columns)
matrix3.size # ==> 12

# View number of dimensions (see also ranks)
# ref: https://stackoverflow.com/questions/19389910/in-python-numpy-what-is-a-dimension-and-axis
# it is the number of axis
print(matrix3.ndim) # ==> 2
##################################################################
# Applying Operations to Elements

# Using matrix2
# & creating a function to add 100 to something
add_100 = lambda i: i + 100

# Create a vectorized function
# "vectorize" converts a function into on etht can apply to every element of an array
# it is essentially a for loop over the elements
vectorized_add_100 = np.vectorize(add_100)

# Apply function to all elements in matrix
vectorized_add_100(matrix2)
# ==> array([[101, 102, 103],
#           [104, 105, 106],
#           [107, 108, 109]])

# can also use:
matrix2 + 100
# for the same result
##################################################################
# Finding the Max and Min values
# Using matrix2

# Return max element
np.max(matrix2) #==> 9

# Return min element
np.min(matrix2)

# Find the max element in each column
np.max(matrix2, axis=0) #==> array([7, 8, 9])

# Find the max element in each row
np.max(matrix2, axis=1) #==> array([3, 6, 9])
##################################################################
# Calculating the avg, Variance and Standard deviation
# Notes: Varience = The expectation of the squared deviation of a random variable from its mean (i.e. it measures how far a set of numbers are spread out from their average value).
#        Standard deviation = The amount of variation or dispersion of a set of data values.

# Using matrix2

# Return mean
np.mean(matrix2) #==> 5.0

# Return variance
np.var(matrix2) #==> 6.666666666667

# Return Standard Deviation
np.std(matrix2) #==> 2.5819888974716112

# Finding the mean value of each column
np.mean(matrix2, axis=0) #==> array([4., 5., 6.])
##################################################################
# Reshaping Arrays
# Create 4X3 matrix
matrix4 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]])

# Reshape matrix into 2X6 matrix
matrix4.reshape(2, 6)
#==> array([[1, 2, 3, 4, 5, 6],
#           [7, 8, 9, 10, 11, 12]])

# To use shape the matrix has to be the same size before and after
# To check size:
matrix4.size #==> 12

# Using -1 in reshape means "as many as needed"
# for example 
matrix4.reshape(1, -1) #==> array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

# Also using a single integer will return a 1D array of that length(has to be the size of the original array)
matrix4.reshape(12) #==> array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
##################################################################
# Transposing a Vector or Matrix
# Using matix2 again

# Transpose matrix
matrix2.T #==> array([[1, 4, 7],
#                     [2, 5, 8],
#                     [3, 6, 9]])

# Technically a vector cannot be transposed so it won't work on one

# It is usually refered to as transposing a row vector to a column vector (using two brackets)
np.array([[1, 2, 3, 4, 5, 6]]).T
#==> array([[1],
#           [2],
#           [3],
#           [4],
#           [5],
#           [6]])

# Flattening a Matrix
# Using matrix2
# Use flatten
matrix2.flatten() #==> array([1, 2, 3, 4, 5, 6, 7, 8, 9])
##################################################################
# Finding the Rank fo a Matrix (http://bit.ly/2HUzkMs , http://bit.ly/2FxSUzC)
# Notes: A 3X5 matrix can only have a max rank of 3 (same with a 5X3)
#        If one row or column (depending on the rank) can be made from a linear combination of others then it is not counted towards the rank
# Create a matrix
matrix5 = np.array([[1, 1, 1],
                    [1, 1, 10],
                    [1, 1, 15]])

# Return matrix rank
np.linalg.matrix_rank(matrix5) #==> 2

# Calculating the Determinant (http://bit.ly/2FA6ToM)
# Using matrix2

# Return the determinant
np.linalg.det(matrix2) #==> 0.0
##################################################################
# Getting the Diagonal of a Matrix (getting the diagonal elements)
# Create matrix
matrix6 = np.array([[1, 2, 3],
                    [2, 4, 6],
                    [3, 8, 9]])

# Return diagonal elements 
matrix6.diagonal() #==> arrat([1, 4, 9])

# Return diagonal one above the main diagonal
matrix6.diagonal(offset=1) #==> arrat([2, 6])

# Return diagonal one above the main diagonal
matrix6.diagonal(offset=-1) #==> arrat([2,8])
##################################################################
# Calculating the Trace of a Matrix (http://bit.ly/2FunM45)
# Sum of diagonal
# Using matrix6
# (same as ==> sum(matrix6.diagonal()))
matrix.trace() #==> 14 
##################################################################
# Finding the Eigenvalues and Eigenvectors (http://bit.ly/2Hb32LV , bit.ly/2HeGppK)
# Create matrix
matrix7 = np.array([[1, -1, 3],
                    [1, 1, 6],
                    [3, 9, 9]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix7)

eigenvalues #==> array([13.55075847, 0.74003145, -3.29078992])

eigenvectors #==> array([[-0.17622017, -0.96677403, -0.53373322],
#                        [-0.435951  ,  0.2053623 , -0.64324848],
#                        [-0.88254925,  0.15223105,  0.54896288]])
##################################################################
# Calculating Dot Products (http://bit.ly/2Fr0AUe)
# Create two vectors
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# Calcluate the dot product
np.dot(vector_a, vector_b) #==> 32

# Also works with:
vector_a @ vector_b #==> 32
##################################################################
# Adding and Subtracting Matrices
# Create 2 matrices
matrix_a = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 2]])

matrix_b = np.array([[1, 3, 1],
                    [1, 3, 1],
                    [1, 3, 8]])

# Add two matrices
np.add(matrix_a, matrix_b)
# or
matrix_a + matrix_b
#==> array([[2, 4, 2],
#           [2, 4, 2],
#           [2, 4, 10]])

# Subtract two matrices
np.subtract(matrix_a, matrix_b)
#==> array([[0, -2, 0],
#           [0, -2, 0],
#           [0, -2, -6]])
##################################################################
# Multiplying two Matrices (bit.ly/2FtpXVr)
# Create two matrices
matrix_c = np.array([[1, 1],
                    [1, 2]])
matrix_d = np.array([[1, 3],
                    [1, 2]])

# Multiply two matrices
np.dot(matrix_c, matrix_d)
# or
matrix_a @ matrix_b
#==> array([[2, 5],
#           [3, 7]])

# To multiply two matrices element-wise
matrix_a * matrix_b
#==> array([[1, 3],
#           [1, 4]])
##################################################################
# Inverting a Matrix (bit.ly/2Fzf0BS)
# Calculting the inverse of a square matrix

# Create a matrix
matrix8 = np.array([[1, 4],
                    [2, 5]])

# Calculate the inverse 
np.linalg.inv(matrix8)
#==> array([[-1.66666667, 1.33333333],
#           [0.66666667, -0.33333333]])

# How to get the identity matrix (AA^-1 = I)
# Multiply matrix and its inverse
matrix8 @ np.linalg.inv(matrix8)
#==> array([[1., 0.],
#           [0., 1.]])
##################################################################
# Generating Random Values
# Set Seed
np.random.seed(0)

# Generate three random floats between 0.0 and 1.0
np.random.random(3)
#==> will output random array with 3 float elements

# Generate three random integers between 1 and 10
np.random.randint(0, 11, 3)
#==> Outputs an array of three random integers

# Draw 3 numbers from a normal distribution with mean 0.0 and std of 1.0
np.random.normal(0.0, 1.0, 3)

# Draw 3 numbers from a logistic distribution with mean 0.0 and std of 1.0
np.random.logistic(0.0, 1.0, 3)

# Draw 3 numbers greater than or equal to 1.0 and less than 2.0
np.random.logistic(1.0, 2.0, 3)

