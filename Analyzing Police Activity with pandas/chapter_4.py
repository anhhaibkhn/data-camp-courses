
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
        pass



def main():
    # chap3 = Visualization()
    # chap3.plotting_table_2(chap3.ri)


    chap4 = Analyzing_effect()  
    # chap4.examining_dataset(chap4.weather)
    # chap4.examining_dataset(chap4.weather)



if __name__ == "__main__":
    main()