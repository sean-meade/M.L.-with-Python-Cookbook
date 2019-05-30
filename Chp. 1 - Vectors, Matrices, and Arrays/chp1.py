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

# Describing a Matrix

# Create matrix
matrix3 = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# View number of rows and columns
matrix3.shape # ==> (3, 4)

# View number of elements (rows * columns)
matrix3.size # ==> 12

# View number of dimensions
# ref: https://stackoverflow.com/questions/19389910/in-python-numpy-what-is-a-dimension-and-axis
# it is the number of axis
print(matrix3.ndim) # ==> 2

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

# Finding the Max and Min values
