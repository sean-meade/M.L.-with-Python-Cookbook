# Chapter 7 - Handling Dates and Times
# 7.0 Introduction
"""
Dates and times (datetimes) are frequently encountered during preprocessing for machine learning, whether the time of 
a particular sale or the year of some public health statistic. In this chapter, we will build a toolbox of strategies 
for handling time series data including tackling time zones and creating lagged time features. Specifically, we will 
focus on the time series tools in the pandas library, which centralizes the functionality of many other libraries.
"""

# 7.1 - Converting Strings to Dates
# When you're given a vector of strings representing dates and times, you want to transform them into time series data
# Using pandas' 'to_datetime' with the format of the date and/or timespecified in the 'format' parameter:

# Load libraries
import numpy as np
import pandas as pd

# Create strings
date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])

# Convert to datetimes
[pd.to_datetime(date, format = '%d-%m-%Y %I:%M %p') for date in date_strings]

# We might also want to add an argument to the errors parameter to handle problems
# Convert to datetimes
[pd.to_datetime(date, format = '%d-%m-%Y %I:%M %p', errors = "coerce") for date in date_strings]

# If 'errors="coerce"', then any problem that occurs will not raise an error (the default behavior) but instead will
# set the value causing the error to NaT (i.e., a missing value).

# Discussion
# When dates and times come as strings, we need to convert them into a data type Python can understand. While there
# are a number of Python tools for converting strings to datetimes, following our use of pandas in other recipes we
# can use 'to_datetime' to conduct the transformation. One obstacle to strings representing dates and times is that 
# the format of the strings can vary significantly between data sources. For example, one vector of dates might 
# represent March 23rd, 2015 as "03-23-15" while another might use "3|23|2015". We can use the 'format' parameter to
# specify the exact format of the string. Here are some common date and time formating codes:

# Complete list of Python String Time Codes (http://strftime.org)
# Code = Description : Example
# %Y = Full year : 2001
# %m = Month with zero pedding : 04
# %d = Day of the month with zero padding : 09
# %I = Hour (12hr clock) w/ zero padding : 02
# %p = AM or PM : AM
# %M = Minute w/ zero padding : 05
# %S = Second w/ zero padding : 09

# 7.2 Handling Time Zones
# When you have time series data and want to add or change time zone information.
# If not specfied, pandas objexts have no time zone. However, we can add a time zone using tz during creation:
# Load library
import pandas as pd

# Create datetime
print(pd.Timestamp('2017-05-01 06:00:00', tz = 'Europe/London'))

# We can add a time zone to a previously created datetime using 'tz_localize':
# Create datetime
date = pd.Timestamp('2017-05-01 06:00:00')

# Set time zone
date_in_london = date.tz_localize('Europe/London')

# Show datetime
print(date_in_london)

# We can also convert to a different time zone:
# Change time zone
print(date_in_london.tz_convert('Africa/Abidjan'))

# Finally, pandas' Series objects can apply 'tz_localize' and 'tz_convert' to every element:
# Create three dates
dates = pd.Series(pd.date_range('2/2/2002', periods = 3, freq = 'M'))

# Set time zone
print(dates.dt.tz_localize('Africa/Abidjan'))

# Discussion:
# pandas supports two sets of strings representing timezones; however, I suggest using pytz library's strings. We 
# can see all theh strings used to represent time zones by importing 'all_timezones':
# Load library
from pytz import all_timezones

# Show two timezones
print(all_timezones[0:2])

# 7.3 Selecting Dates and Times
# When you have a vector of dates and you want to select one or more.
# Use two boolean conditions as the start and end dates:
# Load library
import pandas as pd

# Create data frame
dataframe = pd.DataFrame()

# Create datetimes
dataframe['date'] = pd.date_range('1/1/2001', periods = 100000, freq = 'H')

# Select observations between two datetimes
dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
           (dataframe['date'] <= '2002-1-1 04:00:00')]

# Alternatively, we can set the date column as the DataFrame's index and then slice using 'loc':
# Set index
dataframe = dataframe.set_index(dataframe['date'])

# Select observations between two datetimes
dataframe.loc['2002-1-1 01:00:00' : '2002-1-1 04:00:00']

# Discussion
# Whether, we use boolean conditions or index slicing is situation dependent. If we wanted to do some complex time
# series manipulation, it might be worth the overhead of setting the date column as the index of the DataFrame, but
# if we wanted to do some simple data wrangling, the boolean conditions might be easier.

# 7.4 Breaking up date data into multiple features
# When you have a column of dates and times and you want to create features for year, month, day, hour, and minute:
# Use pandas 'Series.dt''s time properties:
# Load library
import pandas as pd

# Create data frame
dataframe = pd.DataFrame()

# Create five dates
dataframe['date'] = pd.date_range('1/1/2001', periods = 150, freq = 'W')

# Create features for year, month, day, hour, and minute
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

# Show three rows
dataframe.head(3)

# Discussion:
# Sometimes it can be useful to break up a column of dates into components. For example, we might want a feature
# that just includes the yearof the observation or we might want only to consider the month of some observation so
# we can compare them regardless of year.

# 7.5 Calculating the difference between dates
# When you have two datetime features and want to calculate the time between them for each observation.
# Subtract the two date features using pandas:
# Load library
import pandas as pd

# Create data frame
dataframe = pd.DataFrame()

# Create two datetime features
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]

# Calculate duration between features
dataframe['Left'] - dataframe['Arrived']

# Often we will want to remove the days output and keep only the numerical value:
# Calculate duration between feature
pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))

# Discussion: (http://bit.ly/2HOS3sJ)
# There are times when the feature we want is the change (delta) between two points in time. For example, we might 
# have the dates a customer checks in and checks out of a hotel, but the feature we want iss the duration of his stay.
# pandas makes this calculation easy using the 'TimeDelta' data type.

# 7.6 Encoding Days of the Week
# When you have a vector of dates and want to know the day of the week for each date.
# Use pandas' 'Series.dt' property 'weekday_name':
# Load library
import pandas as pd

# Create dates
dates = pd.Series(pd.date_range("2/2/2002", periods = 3, freq = "M"))

# Show days of the week
dates.dt.weekday_name

# If we want the output to be a numerical value and therefore more usable as a machine learning feature, we can use
# 'weekday' where the days of the week are represented as an integer (Monday is 0):
# Show days of the week
dates.dt.weekday

# Discussion: (http://bit.ly/2HShhq1)
# Knowing the weekday can be helpful if, for instance, we wanted to compare total sales on Sundays for the past three 
# years. pandas makes creating a feature vector containing weekday information easy.

# 7.7 Creating a Lagged Feature
# When you want to create a feature that is lagged n time periods.
# Use pandas' 'shift'
# Load library
import pandas as pd

# Create data frame
dataframe = pd.DataFrame()

# Create data
dataframe["date"] = pd.date_range("1/1/2001", periods = 5, freq = "D")
dataframe["stock_price"] = [1.1, 2.2, 3.3, 4.4, 5.5]

# Lagged values by one row
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)

# Show data frame
print(dataframe)

# Discussion:
# Very often data is based on regularly spaced time periods (e.g., every day, every hour, every three hours) and we
# are interested in using values in the past to make predictions (this is often called lagging a feature). For 
# example, we might want to predict a stock's price using the price it was the day before. With pandas we can use 
# 'shift' to lag values by one row, creating a new feature containing past values.

# In our solution, the first row for 'previous_days_stock_prices' is a missing value because there is no previous
# 'stock_price' value.

# 7.8 Using Rolling Time Windows
# When you're given time series data, you want to calculate some statistic for a rolling time.

# Load library
import pandas as pd

# Create datetimes
time_index = pd. date_range("01/01/2010", periods = 5, freq = "M")

# Create data frame, set index
dataframe = pd.DataFrame(index = time_index)

# Create feature
dataframe["Stock_Price"] = [1, 2, 3, 4, 5]

# Calculate rolling mean
dataframe.rolling(window = 2).mean()

# Discussion: (http://bit.ly/2HTIicA & http://bit.ly/2HRRHBn)
# Rolling (also called moving) time windows are conceptually simple but can be difficult to understand at first. 
# Imagine we have monthly observations for a stock's price. It is often useful to have a time window of a certain
# number of months and then move over the observations calculating a statistic for all observations in the time 
# window.

# For example, if we have a time window of three months and we want a rolling mean, we would calculate:

# 1 - mean(January, February, March)
# 2 - mean(February, March, April)
# 3 - mean(March, April, May)
# 4 - etc.

# Another way to put it: our three-month time window "walks" over the observations, calculating the window's mean at
# each step.

# pandas' 'rolling' allows us to specify the size of the window using 'window' and then quickly calculate some common
# statistics, including the max value (max()), mean value (mean()), count of values (count()), and rolling correlation
# (corr()).

# Rolling means are often used to smooth out times series date because using the mean of the entire time window 
# dampens the effect of short-term fluctuations.

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
dataframe.ffill()

# We cab also replace missing values with the latest known value (i.e., back-filling):
# Back-fill
dataframe.bfill()

# Discussion:
# Interpolation is a technique for filling in gaps caused by missing values by, in effect, drawing a line or curve
# to predict reasonable values. Interpolation can be particularly useful when the time intervals between are constant,
# the data is not prone to noisy fluctuations, and the gaps caused by missing values are small. For example, in our
# solution a gap of two missing values was bordered by 2.0 and 5.0. By fitting a line starting at 2.0 and ending at 
# 5.0, we can make reasonable guesses for the two missing values in between of 3.0 and 4.0.

# If we believe the line between the two known points is nonlinear, we can use 'interpolare''s 'method' to specify
# the interpolation method:

# Interpolate missing values
dataframe.interpolate(method = "quadratic")

# Finally, there might be cases when we have large gaps of missing values and do not want to interpolate values across
# the entire gap. In these cases we can use 'limit' to restrict the number of interpolated values and 'limit_direction'
# to set whether to interpolate values forward from at the last known value before the gap or vice versa:

# Interpolate missing values
dataframe.interpolate(limit = 1, limit_direction = "forward")

# Back-filling and forward-filling can be thought of as a form of naive interpolation, where we draw a flat line from 
# a known value and use it to fill in missing values. One (minor) advantage back- and forward-filling have over 
# interpolation is the lack of the need for known values on both sides of missing value(s).