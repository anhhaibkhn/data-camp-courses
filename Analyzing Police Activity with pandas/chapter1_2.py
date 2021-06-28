""" Course Outline: Now that you have learned the foundations of pandas, this course will give you the chance to apply that knowledge by answering interesting questions 
about a real dataset! You will explore the Stanford Open Policing Project dataset and analyze the impact of gender on police behavior. 
During the course, you will gain more practice cleaning messy data, creating visualizations, combining and reshaping datasets, and manipulating time series data. 
Analyzing Police Activity with pandas will give you valuable experience analyzing a dataset from start to finish, preparing you for your data science career! 
handles dates, common date operations, and the right way to format dates to avoid confusion.
"""

""" Chapter 1: Preparing the data for analysis
Before beginning your analysis, it is critical that you first examine and clean the dataset, to make working with it a more efficient process. In this chapter, 
you will practice fixing data types, handling missing values, and dropping columns and rows while learning about the Stanford Open Policing Project dataset.

"""
import pandas as pd
import numpy as np

class Preparing_data():

    def __init__(self):
        self.show_data = False
        # Rhode island
        self.ri = pd.read_csv("police.csv")
        # self.ri = self.summary_df(self.ri, self.show_data )

        # Drop the 'county_name' and 'state' columns
        self.ri.drop(['county_name', 'state'], axis='columns', inplace=True)

        if self.show_data:
            # Examine the shape of the DataFrame (again)
            print(self.ri.shape)

            # Examine the head of the 'is_arrested' column
            print(self.ri.is_arrested.head())

        # Change the data type of 'is_arrested' to 'bool'
        self.ri['is_arrested'] = self.ri.is_arrested.astype('bool')

        # Check the data type of 'is_arrested' 
        # print(self.ri.is_arrested.dtypes)

        self.ri.dropna(how='all')
        
        

        
    def summary_df(self,df, show = False):
        # locating missing values, count the missing values in each col
        if show:
            print("# locating missing values, count the missing values in each col")
            print(df.isnull().sum())
            print("# data shape: ", df.shape)
            print("# drop na, drop duplicates")

        df.dropna(how='all')
        df.drop_duplicates()
        if show:
            print("# data shape: ", df.shape)
            print("# Summary of the basic information about this DataFrame and its data:")
            print(df.info())
            print("# Top 5 rows:")
            print(df.head())
            print("# Statistical data:")
            print(df.describe())
        
        return df


    def exercise_1(self, ri):
        """ Counting events per calendar month  """
        # Concatenate 'stop_date' and 'stop_time' (separated by a space)
        combined = ri.stop_date.str.cat(ri.stop_time, sep = ' ')

        # Convert 'combined' to datetime format
        ri['stop_datetime'] = pd.to_datetime(combined)

        # Examine the data types of the DataFrame
        print(ri.dtypes)

        # Set 'stop_datetime' as the index
        print("set ri index as stop_time")
        ri.set_index('stop_datetime', inplace=True)

        # Examine the index
        print(ri.index)

        # Examine the columns
        print("### examine the columns")
        print(ri.columns)
        print("# Finished ")

        return ri

    
    

""" Chapter 2 Exploring the relationship between gender and policing
Does the gender of a driver have an impact on police behavior during a traffic stop? 
In this chapter, you will explore that question while practicing filtering, grouping, method chaining, Boolean math, string methods, and more!  """

class Exploring(Preparing_data):

    def __init__(self):
        super().__init__()

        # creating onebike_datetimes dict
        


    def exercise_1(self, ri):
        """ Counts the unique values """
        print(ri.stop_outcome.value_counts())
        print("# Total counts: ",ri.stop_outcome.value_counts().sum())
        print("# count with nomarlization")
        print(ri.stop_outcome.value_counts(normalize = True))
        
        print(ri.driver_race.value_counts())
        white = ri[ri.driver_race == 'White']
        print(white.shape)
        print("# White with normalization", "\n", \
                 white.stop_outcome.value_counts(normalize = True))

        asian = ri[ri.driver_race == 'Asian']
        print(asian.stop_outcome.value_counts(normalize = True))

    def exercise_2(self, ri):
        # Count the unique values in 'violation'
        print("# Total violation counts: ")
        print(ri.violation.value_counts().sum())

        # Express the counts as proportions
        print(ri.violation.value_counts(normalize = True))

        # Create a DataFrame of female drivers
        female = ri[ri.driver_gender == 'F']

        # Create a DataFrame of male drivers
        male = ri[ri.driver_gender == 'M']

        print("# Female violation counts: ")
        print(female.violation.value_counts().sum())
        # Compute the violations by female drivers (as proportions)
        print(female.violation.value_counts(normalize = True))

        print("# Male violation counts: ")
        print(male.violation.value_counts().sum())
        # Compute the violations by male drivers (as proportions)
        print(male.violation.value_counts(normalize = True))

    def exercise_3(self, ri):
        # Create a DataFrame of female drivers stopped for speeding
        female_and_speeding = ri[(ri.driver_gender == 'F') & (ri.violation == 'Speeding')]

        # Create a DataFrame of male drivers stopped for speeding
        male_and_speeding = ri[(ri.driver_gender == 'M') & (ri.violation == 'Speeding')]

        # Compute the stop outcomes for female drivers (as proportions)
        print(female_and_speeding.stop_outcome.value_counts(normalize = True))

        # Compute the stop outcomes for male drivers (as proportions)
        print(male_and_speeding.stop_outcome.value_counts(normalize = True)) 


    def ex_4(self, ri):
        # Math with boolean values
        print(ri.isnull().sum())
        print("# np.mean([0,1,0,0]): ", np.mean([0,1,0,0]))
        """ The mean of the Boolean series represent the percentage of the True values"""

        # Taking the mean of a Boolean series
        print(ri.is_arrested.value_counts(normalize = True))
        print(ri.is_arrested.mean())
        print(ri.is_arrested.dtype)

        # similarly for search_conducted column 
        # Check the data type of 'search_conducted'
        print(ri.search_conducted.dtype)

        # Calculate the search rate by counting the values
        print(ri.search_conducted.value_counts(normalize = True))

        # Calculate the search rate by taking the mean
        print(ri.search_conducted.mean())

    def ex_5(self,ri):
        """ Comparing groups using groupby """
        # Study the arrest rate by the police district 
        print(ri.district.unique())

        # filetring K1 
        print(ri[ri.district == 'Zone K1'].is_arrested.mean())

        # Using groupby 
        print(ri.groupby('district').is_arrested.mean())

        # using gropby for multiple category
        print(ri.groupby(['district', 'driver_gender']).is_arrested.mean())

        # using gropby for multiple category with REVERSE order
        print(ri.groupby(['driver_gender', 'district']).is_arrested.mean())

    def ex_6(self,ri):
        # Count the 'search_type' values
        print(ri.search_type.value_counts())

        # Check if 'search_type' contains the string 'Protective Frisk'
        ri['frisk'] = ri.search_type.str.contains('Protective Frisk', na=False)

        # Check the data type of 'frisk'
        print(ri['frisk'].dtypes)

        # Take the sum of 'frisk'
        print(ri['frisk'].sum())

        # Create a DataFrame of stops in which a search was conducted
        searched = ri[ri.search_conducted == True]
        searched.head()

        # Calculate the overall frisk rate by taking the mean of 'frisk'
        print(searched.frisk.mean())

        # Calculate the frisk rate for each gender
        print(searched.groupby('driver_gender').frisk.mean())





def main():

    chap1 = Preparing_data()  
    # chap1.exercise_1(chap1.ri)
    chap2 = Exploring()
    chap2.ex_4(chap2.ri)
    chap2.ex_5(chap2.ri)

if __name__ == "__main__":
    main()