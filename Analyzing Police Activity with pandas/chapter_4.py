
from chapter1_2 import Preparing_data
import pandas as pd
import matplotlib.pyplot as plt

"""
Chapter 3: Visual exploratory data analysis

Are you more likely to get arrested at a certain time of day? Are drug-related stops on the rise? 
In this chapter, you will answer these and other questions by analyzing the dataset visually, 
since plots can help you to understand trends in a way that examining the raw data cannot
"""
class Visualization(Preparing_data):
    def __init__(self):
        super().__init__()

        self.table = pd.crosstab(self.ri.driver_race, self.ri.driver_gender)


    def plotting_table(self, ri):
        # computing a frequence table, how many times each combination of values occurs
        table = pd.crosstab(ri.driver_race, ri.driver_gender)
        print(table)

        # verify the shape 
        print(ri[(ri.driver_race == 'Asian') & (ri.driver_gender == 'F')].shape)

        # overwrite table to a smaller table 
        table = table.loc['Asian':'Hispanic']

        # creating a line plot
        table.plot()
        plt.show()

        # creating a bar plot, with more appropriate
        table.plot(kind = 'bar')
        plt.show()

        # creating stacking bars plot
        table.plot( kind = 'bar', stacked = True)
        plt.show()

    def plotting_table_2(self, ri):
        # Create a frequency table of districts and violations
        print(pd.crosstab(ri.district, ri.violation))

        # Save the frequency table as 'all_zones'
        all_zones = pd.crosstab(ri.district, ri.violation)

        # Select rows 'Zone K1' through 'Zone K3'
        print(all_zones.loc['Zone K1':'Zone K3'])

        # Save the smaller table as 'k_zones'
        k_zones = all_zones.loc['Zone K1':'Zone K3'] 

        # Create a bar plot of 'k_zones'
        k_zones.plot(kind = 'bar')
        # Display the plot
        plt.show()

        # Create a stacked bar plot of 'k_zones'
        k_zones.plot(kind = 'bar', stacked = True)
        # Display the plot
        plt.show()

    def sort_or_rotate_plots(self, ri):
        # Print the unique values in 'stop_duration'
        print(ri.stop_duration.unique())

        # Create a dictionary that maps strings to integers
        mapping = {'0-15 Min':8, '16-30 Min':23,'30+ Min':45}

        # Convert the 'stop_duration' strings to integers using the 'mapping'
        ri['stop_minutes'] = ri.stop_duration.map(mapping)

        # Print the unique values in 'stop_minutes'
        print(ri.stop_minutes.unique())

        # Calculate the mean 'stop_minutes' for each value in 'violation_raw'
        print(ri.groupby(ri.violation_raw).stop_minutes.mean())

        # Save the resulting Series as 'stop_length'
        stop_length = ri.groupby(ri.violation_raw).stop_minutes.mean()

        # Sort 'stop_length' by its values and create a horizontal bar plot
        stop_length.sort_values().plot(kind = 'barh')

        # Display the plot
        plt.show()


"""
Chapter 4: Analyzing the effect of weather on policing

In this chapter, you will use a second dataset to explore the impact of weather conditions on police behavior 
during traffic stops. You will practice merging and reshaping datasets, assessing whether a data source is 
trustworthy, working with categorical data, and other advanced skills.

"""

class Analyzing_effect(Preparing_data):

    def __init__(self):
        super().__init__()

        # creating onebike_datetimes dict
        self.weather = pd.read_csv('weather.csv')
        

    def examining_dataset(self, weather):
        # examining the wind speed
        print(weather[['AWND', 'WSF2']].head())
        print(weather[['AWND', 'WSF2']].describe())

        # examniinig with box plot
        weather[['AWND', 'WSF2']].plot(kind = 'box')
        plt.show()

        # examiniing with histogram
        weather['WDIFF'] = weather.WSF2 - weather.AWND
        weather.WDIFF.plot(kind = 'hist', bins = 20)
        plt.show()

    def examining_dataset2(self, weather):

        # Describe the temperature columns
        print(weather[['TMIN', 'TAVG', 'TMAX']].describe())

        # Create a box plot of the temperature columns
        weather[['TMIN', 'TAVG', 'TMAX']].plot(kind='box')
        # Display the plot
        plt.show()

        # Create a 'TDIFF' column that represents temperature difference
        weather['TDIFF'] = weather.TMAX - weather.TMIN

        # Describe the 'TDIFF' column
        print(weather['TDIFF'].describe())

        # Create a histogram with 20 bins to visualize 'TDIFF'
        weather.TDIFF.plot(kind = 'hist', bins = 20)
        # Display the plot
        plt.show()

    def categorizing_weather(self, weather):
        # selecting dataframe slice 
        temp = weather.loc[:, 'TAVG': 'TMAX']
        print(temp.shape, '\n', temp.columns, '\n','######################')

        # check dataframe operations
        print(temp.head(), '\n', temp.sum(), '\n','#######################')

        print(temp.sum(axis = 'columns'), '\n','#######################')
    
    def mapping_values(self, df):
        ri = df.dropna()
        # mapping one set of values to another 
        print(ri.stop_duration.unique(), '\n','#######################', ri.stop_duration.shape)

        # mapping dict
        mapping = { '0-15 Min': 'short',\
                    '16-30 Min': 'medium',\
                    '30+ Min': 'long' }
        # creating new column stop_length (string data) with mapping dict
        ri["stop_length"] = ri['stop_duration'].map(mapping)
        print(ri.stop_length.unique(), '\n','#######################')
        
        #change it to category type, for more efficiently storing the data
        print(ri.stop_length.memory_usage(deep = True), '\n','#######################')

        cats = ['short', 'medium', 'long']
        # ri['stop_length'] = ri.stop_length.astype('category', categories = cats)
        ri['stop_length'] = pd.Categorical( ri.stop_length, categories=cats, ordered=True)
        print('### after changing to category type')
        print(ri.stop_length.memory_usage(deep = True), '\n','#######################')
        print(ri.stop_length.head())

        # using the ordered categories
        print(ri[ri.stop_length > 'short'].shape)

    def counting_values(self, weather):
        # Copy 'WT01' through 'WT22' to a new DataFrame
        WT = weather.loc[:,'WT01':'WT22']

        weather.head()

        # Calculate the sum of each row in 'WT'
        weather['bad_conditions'] = WT.sum(axis = 'columns')

        # Replace missing values in 'bad_conditions' with '0'
        weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')

        # Create a histogram to visualize 'bad_conditions'
        weather['bad_conditions'].plot(kind = 'hist')
        # Display the plot
        plt.show()

        # plotting_bad_conditions_weathe
        # Count the unique values in 'bad_conditions' and sort the index
        print(weather.bad_conditions.value_counts().sort_index())

        # Create a dictionary that maps integers to strings
        # creating dic that contains multiple key having same value
        mapping = {0:'good', 1:'bad', 2:'bad', **dict.fromkeys([3,4], 'bad'), \
                                             **dict.fromkeys([5,6,7,8,9], 'worse')}

        # Convert the 'bad_conditions' integers to strings using the 'mapping'
        weather['rating'] = weather.bad_conditions.map(mapping)

        # Count the unique values in 'rating'
        print(weather.rating.value_counts())

        # Create a list of weather ratings in logical order
        cats = ['good', 'bad', 'worse']

        # Change the data type of 'rating' to category
        # weather['rating'] = weather.rating.astype('category', ordered=True, categories = cats)
        weather['rating'] = pd.Categorical( weather.rating, categories=cats, ordered=True)

        # Examine the head of 'rating'
        print(weather.rating.head())
        return weather

    def merge_df(self, ri, weather):

        # Reset the index of 'ri'
        ri.reset_index(inplace=True)

        # Examine the head of 'ri'
        print(ri.head())

        # Create a DataFrame from the 'DATE' and 'rating' columns
        weather_rating = weather[['DATE', 'rating']]

        # Examine the head of 'weather_rating'
        print(weather_rating.head())

        # Examine the shape of 'ri'
        print(ri.shape)

        # Merge 'ri' and 'weather_rating' using a left join
        ri_weather = pd.merge(left=ri, right=weather_rating, left_on='stop_date', right_on='DATE', how='left')

        # Examine the shape of 'ri_weather'
        print(ri_weather.shape)

        # Set 'stop_datetime' as the index of 'ri_weather'
        ri_weather.set_index('stop_datetime', inplace=True)

        return ri_weather

    def comparing_arrest_rate(self, ri_weather):
        # Calculate the overall arrest rate
        print(ri_weather.is_arrested.mean())

        # Calculate the arrest rate for each 'rating'
        print(ri_weather.groupby('rating').is_arrested.mean())

        # Calculate the arrest rate for each 'violation' and 'rating'
        print(ri_weather.groupby(['violation', 'rating']).is_arrested.mean())

        # Save the output of the groupby operation from the last exercise
        arrest_rate = ri_weather.groupby(['violation', 'rating']).is_arrested.mean()

        # Print the 'arrest_rate' Series
        print(arrest_rate)

        # Print the arrest rate for moving violations in bad weather
        print(arrest_rate.loc['Moving violation','bad'])

        # Print the arrest rates for speeding violations in all three weather conditions
        print(arrest_rate.loc['Speeding'])

        # Unstack the 'arrest_rate' Series into a DataFrame
        print(arrest_rate.unstack)

        # Create the same DataFrame using a pivot table
        print(ri_weather.pivot_table(index='violation', columns='rating', values='is_arrested'))
        



def main():
    # chap3 = Visualization()
    # chap3.plotting_table_2(chap3.ri)


    chap4 = Analyzing_effect()  
    # chap4.examining_dataset(chap4.weather)
    # chap4.examining_dataset(chap4.weather)
    # chap4.categorizing_weather(chap4.weather)
    # chap4.mapping_values(chap4.ri)
    new_ri = chap4.exercise_1(chap4.ri)
    new_weather = chap4.counting_values(chap4.weather)
    ri_weather = chap4.merge_df(new_ri, new_weather)

    chap4.comparing_arrest_rate(ri_weather)






if __name__ == "__main__":
    main()