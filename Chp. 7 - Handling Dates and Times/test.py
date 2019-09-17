

# 7.9 Handling Missing Data in Time Series
# When you have missing values in time series data.
# In addition to the missing strategies previously discussed, when we have time series data we can use interpolation
# to fill in gaps caused by missing values:
# Load libraries
import pandas as pd
import numpy as np

# Create date
time_index = pd.date_range("01/01/2010", periods = 5, freq = "M")

# Create data frame, set index
dataframe = pd.DataFrame(index = time_index)

# Create feature with a gap of missing values
dataframe["Sales"] = [1.0, 2.0, np.nan, np.nan, 5.0]

# Interpolate missing values
print(dataframe.interpolate())

# Alternatively, we can replace missing values with the last known value (i.e., forward-filling):
# Forward-fill
print(dataframe.ffill())

# We cab also replace missing values with the latest known value (i.e., back-filling):
# Back-fill
print(dataframe.bfill())

# Discussion:
# Interpolation is a technique for filling in gaps caused by missing values by, in effect, drawing a line or curve
# to predict reasonable values. Interpolation can be particularly useful when the time intervals between are constant,
# the data is not prone to noisy fluctuations, and the gaps caused by missing values are small. For example, in our
# solution a gap of two missing values was bordered by 2.0 and 5.0. By fitting a line starting at 2.0 and ending at 
# 5.0, we can make reasonable guesses for the two missing values in between of 3.0 and 4.0.

# If we believe the line between the two known points is nonlinear, we can use 'interpolare''s 'method' to specify
# the interpolation method:

# Interpolate missing values
print(dataframe.interpolate(method = "quadratic"))

# Finally, there might be cases when we have large gaps of missing values and do not want to interpolate values across
# the entire gap. In these cases we can use 'limit' to restrict the number of interpolated values and 'limit_direction'
# to set whether to interpolate values forward from at the last known value before the gap or vice versa:

# Interpolate missing values
print(dataframe.interpolate(limit = 1, limit_direction = "forward"))
