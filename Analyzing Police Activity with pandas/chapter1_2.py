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

class Preparing_data():

    def __init__(self):
        self.show_data = True
        # Rhode island
        self.ri = pd.read_csv("police.csv")
        # self.ri = self.summary_df(self.ri, self.show_data )

        # Drop the 'county_name' and 'state' columns
        self.ri.drop(['county_name', 'state'], axis='columns', inplace=True)

        # Examine the shape of the DataFrame (again)
        print(self.ri.shape)

        # Examine the head of the 'is_arrested' column
        print(self.ri.is_arrested.head())

        # Change the data type of 'is_arrested' to 'bool'
        self.ri['is_arrested'] = self.ri.is_arrested.astype('bool')

        # Check the data type of 'is_arrested' 
        print(self.ri.is_arrested.dtypes)
        

        
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
        ri.set_index('stop_datetime', inplace=True)

        # Examine the index
        print(ri.index)

        # Examine the columns
        print(ri.columns)

    
    

""" Chapter 2 Exploring the relationship between gender and policing
Does the gender of a driver have an impact on police behavior during a traffic stop? 
In this chapter, you will explore that question while practicing filtering, grouping, method chaining, Boolean math, string methods, and more!  """

class Exploring(Preparing_data):

    def __init__(self):
        super().__init__()

        # creating onebike_datetimes dict
        self.onebike_datetimes = []
        start_list = self.bike_data_df['Start date'].to_list()
        end_list =  self.bike_data_df['End date'].to_list()
        


    def exercise_1(self, onebike_datetimes):
        """ Counting events before and after noon """
        # Create dictionary to hold results
        trip_counts = {'AM': 0, 'PM': 0}
        
        # Loop over all trips
        for trip in onebike_datetimes:
            # Check to see if the trip starts before noon
            if trip['start'].hour < 12:
                # Increment the counter for before noon
                trip_counts['AM'] += 1
            else:
                # Increment the counter for after noon
                trip_counts['PM'] += 1
        
        print(trip_counts)




def main():

    chap1 = Preparing_data()  
    chap1.exercise_1(chap1.ri)

if __name__ == "__main__":
    main()