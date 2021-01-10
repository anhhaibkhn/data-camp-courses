
''' Chapter 1:    Read, clean, and validate
The first step of almost any data project is to read the data, 
check for errors and special cases, and prepare data for analysis. 
This is exactly what you'll do in this chapter, 
while working with a dataset obtained from the National Survey of Family Growth.'''
# 1 Exploring the NSFG data
'''
To get the number of rows and columns in a DataFrame, you can read its shape attribute.
To get the column names, you can read the columns attribute. 
The result is an Index, which is a Pandas data structure that is similar to a list. '''

import pandas as pd
import numpy as np

paras_verbose = 1
def verbose(msg, detail_msg =''):
    ''' Verbose function for print information to stdout'''
    if paras_verbose == 1:
        print('[INFO]', msg, detail_msg)

nsfg = pd.read_hdf('nsfg.hdf5')
# Display the number of rows and columns
verbose('nsfg.shape',nsfg.shape)
verbose('nsfg.columns',nsfg.columns)

# 2 Clean and Validate
# each column is a series 
pounds = nsfg['birthwgt_lb1']
ounces = nsfg['birthwgt_oz1']
verbose('type(pounds)',type(pounds)) 

# validate data, sort the series 
pounds.value_counts().sort_index() 
# describe 
pounds.describe() # mean = 8.055 
# replace 99pounds = 44.9056 kg, invalid data for babe weight
pounds = pounds.replace([98, 99], np.nan)
verbose('pounds.mean()',pounds.mean())  # updated mean = 6.703286384976526
# ounces.replace([98, 99], np.nan, inplace=True)  using modifying the existing series in place 

# 3 Practice
# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference (pregnancy length)
preg_length = agepreg - agecon

# 4  Compute summary statistics
verbose('compute the mean duration from conception to end of pregnancy',preg_length.describe())

# 5 Plot the histogram
import matplotlib.pyplot as plt
verbose('Plotting the histogram')
plt.hist(agecon, bins=20, histtype='step')

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')
# Show the figure
plt.show()

#6  Compute birth weight

def resample_rows_weighted(df, column='wgt2013_2015'):
    """Resamples a DataFrame using probabilities proportional to given column.
    Args:
        df: DataFrame
        column: string column name to use as weights
    returns: 
        DataFrame
    """
    weights = df[column].copy()
    weights /= sum(weights)
    indices = np.random.choice(df.index, len(df), replace=True, p=weights)
    sample = df.loc[indices]
    return sample

# Resample the data
nsfg = resample_rows_weighted(nsfg, 'wgt2013_2015')

# Clean the weight variables
pounds = nsfg['birthwgt_lb1'].replace([98, 99], np.nan)
ounces = nsfg['birthwgt_oz1'].replace([98, 99], np.nan)

# Compute total birth weight
birth_weight = pounds + ounces/16

verbose(' Compute the full term birth weight')
# Create a Boolean Series for full-term babies
full_term = nsfg['prglngth'] >= 37

# Select the weights of full-term babies
full_term_weight = birth_weight[full_term]

# Compute the mean weight of full-term babies
print(full_term_weight.mean())


# Filter full-term babies
verbose(' Filter full-term babies')
full_term = nsfg['prglngth'] >= 37

# Filter single births
single = nsfg['nbrnaliv'] == 1

# Compute birth weight for single full-term babies
single_full_term_weight = birth_weight[full_term & single]
print('Single full-term mean:', single_full_term_weight.mean())

# Compute birth weight for multiple full-term babies
mult_full_term_weight = birth_weight[full_term & (~ single)]
print('Multiple full-term mean:', mult_full_term_weight.mean())