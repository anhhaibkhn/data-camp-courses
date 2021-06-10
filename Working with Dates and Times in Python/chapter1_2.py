""" Chapter 1  Dates and Calendars
Hurricanes (also known as cyclones or typhoons) hit the U.S. state of Florida several times per year. 
To start off this course, you'll learn how to work with date objects in Python, 
starting with the dates of every hurricane to hit Florida since 1950. 
You'll learn how Python handles dates, common date operations, and the right way to format dates 
to avoid confusion.

"""
from datetime import date, datetime
import pandas as pd
import numpy as np

class Date_Calendars():

    def __init__(self):
        self.show_data = True
        self.bike_data_df = pd.read_csv("capital-onebike.csv")
        self.bike_data_df = self.summary_df(self.bike_data_df, self.show_data )

        self.florida_hurricane = pd.read_pickle("florida_hurricane_dates.pkl")
        print(type(self.florida_hurricane), self.florida_hurricane[:5])

    def summary_df(self,df, show = False):
        df.dropna(how='all')
        df.drop_duplicates()
        if show:
            print("Summary of the basic information about this DataFrame and its data:")
            print(df.info())
            print("Top 5 rows:")
            print(df.head())
            print("Statistical data:")
            print(df.describe())
        
        return df


    def exercise_1(self, df):
        """ Counting events per calendar month  """
        florida_hurricane_dates = df
        # A dictionary to count hurricanes per calendar month
        hurricanes_each_month = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:0,
                                7: 0, 8:0, 9:0, 10:0, 11:0, 12:0}

        # Loop over all hurricanes
        for hurricane in florida_hurricane_dates:
            # Pull out the month
            month = hurricane.month
            # Increment the count in your dictionary by one
            hurricanes_each_month[month] += 1
            
        print(hurricanes_each_month)
    
    def exercise_2(self, df):
        florida_hurricane_dates = df
        # Assign the earliest date to first_date
        first_date = min(florida_hurricane_dates)

        # Convert to ISO and US formats
        iso = "Our earliest hurricane date: " + first_date.isoformat()
        us = "Our earliest hurricane date: " + first_date.strftime("%m/%d/%Y")

        print("ISO: " + iso)
        print("US: " + us)


""" Chapter 2 Combining Dates and Times
Bike sharing programs have swept through cities around the world -- and luckily for us, 
every trip gets recorded! Working with all of the comings and goings of one bike in Washington, D.C.,
 you'll practice working with dates and times together, parse dates and times from text, 
 analyze peak trip times, calculate ride durations, and more."""

class Combine_date_time(Date_Calendars):

    def __init__(self):
        super().__init__()

        # creating onebike_datetimes dict
        self.onebike_datetimes = []
        start_list = self.bike_data_df['Start date'].to_list()
        end_list =  self.bike_data_df['End date'].to_list()
        print(start_list[:2])
        for s,e in zip(start_list, end_list):
             self.onebike_datetimes.append({'start': datetime.strptime(s,"%Y-%m-%d %H:%M:%S"), \
                                              'end': datetime.strptime(e,"%Y-%m-%d %H:%M:%S")})
        print(self.onebike_datetimes[:2])


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

    # chap1 = Date_Calendars()  
    # chap1.exercise_1(chap1.florida_hurricane) 
    # chap1.exercise_2(chap1.florida_hurricane) 

    chap2 = Combine_date_time()
    chap2.exercise_1(chap2.onebike_datetimes)


if __name__ == "__main__":
    main()