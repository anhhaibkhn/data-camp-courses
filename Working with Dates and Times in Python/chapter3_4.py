""" Chapter 3: Time Zones and Daylight Saving

In this chapter, you'll learn to confidently tackle the time-related topic that causes people the most trouble:
 time zones and daylight saving. Continuing with our bike data, you'll learn how to compare clocks around the world,
 how to gracefully handle "spring forward" and "fall back," and how to get up-to-date timezone data from the dateutil library."""

# Import datetime, timedelta, tz, timezone
from datetime import datetime, timedelta, timezone
from dateutil import tz
from chapter1_2 import Combine_date_time
from IPython.display import display

class Timezones_Daylight(Combine_date_time):

    def __init__(self):
        super().__init__()


    def chap3_ex1(self):
        """ How many hours elapsed around daylight saving? """
        # Start on March 12, 2017, midnight, then add 6 hours
        start = datetime(2017, 3, 12, tzinfo = tz.gettz('America/New_York'))
        end = start + timedelta(hours=6)
        print(start.isoformat() + " to " + end.isoformat())

        # How many hours have elapsed?
        print((end - start).total_seconds()/(60*60))

        # What if we move to UTC?
        print((end.astimezone(tz.UTC) - start.astimezone(tz.UTC))\
            .total_seconds()/(60*60))

    def chap3_ex2(self):
        """ March 29, throughout a decade """
        # Create starting date
        dt = datetime(2000, 3, 29, tzinfo = tz.gettz('Europe/London'))

        # Loop over the dates, replacing the year, and print the ISO timestamp
        for y in range(2000, 2011):
            print(dt.replace(year=y).isoformat())

    def chap3_ex4(self, df):
        # 1 Finding ambiguous datetimes
        onebike_datetimes =  df
        eastern = tz.gettz('US/Eastern')   
        # Loop over trips
        for trip in onebike_datetimes:
            # need to specify the time zone to US/Eastern firs
            start = trip['start'].replace(tzinfo=eastern)
            end = trip['end'].replace(tzinfo=eastern)
            # Rides with ambiguous start
            if tz.datetime_ambiguous(start):
                print("Ambiguous start at " + str(start))
            # Rides with ambiguous end
            if tz.datetime_ambiguous(end):
                print("Ambiguous end at " + str(end))

    def chap3_ex5(self, onebike_datetimes):
        # this one needs the dataset with the right timezone
        # 2 Cleaning daylight saving data with fold
        trip_durations = []
        for trip in onebike_datetimes:
            # When the start is later than the end, set the fold to be 1
            if trip['start'] > trip['end']:
                trip['end'] = tz.enfold(trip['end'])
            # Convert to UTC
            start = trip['start'].astimezone(tz.UTC)
            end = trip['end'].astimezone(tz.UTC)

            # Subtract the difference
            trip_length_seconds = (end-start).total_seconds()
            trip_durations.append(trip_length_seconds)

        # Take the shortest trip duration
        print("Shortest trip: " + str(min(trip_durations)))

    def chap4_ex1(self,rides):
        # display(df.head(3))
        # Subtract the start date from the end date
        ride_durations = rides['End date'] - rides['Start date']

        # Convert the results to seconds
        rides['Duration'] = ride_durations.dt.total_seconds()

        print(rides['Duration'].head())

        """Suppose you have a theory that some people take long bike rides before putting their bike back in the same dock. 
        Let's call these rides "joyrides"."""
        # Create joyrides
        joyrides = (rides['Start station'] == rides['End station'])

        # Total number of joyrides
        print("{} rides were joyrides".format(joyrides.sum()))

        # Median of all rides
        print("The median duration overall was {:.2f} seconds"\
            .format(rides['Duration'].median()))

        # Median of joyrides
        print("The median duration for joyrides was {:.2f} seconds"\
            .format(rides[joyrides]['Duration'].median()))

    def chap4_ex2(self, rides):
        # Resample rides to be monthly on the basis of Start date
        monthly_rides = rides.resample('M', on = 'Start date')['Member type']

        # Take the ratio of the .value_counts() over the total number of rides
        print(monthly_rides.value_counts() / monthly_rides.size())

        # Group rides by member type, and resample to the month
        grouped = rides.groupby('Member type')\
        .resample('M', on = 'Start date')

        # Print the median duration for each group
        print(grouped['Duration'].median())
    
    def chap4_ex3(self, rides):
        # Localize the Start date column to America/New_York
        rides['Start date'] = rides['Start date'].dt.tz_localize('America/New_York', 
                                                                ambiguous='NaT')

        # Print first value
        print(rides['Start date'].iloc[0])

        # Convert the Start date column to Europe/London
        rides['Start date'] = rides['Start date'].dt.tz_convert('Europe/London')

        # Print the new value
        print(rides['Start date'].iloc[0])

    def chap4_ex4(self, rides):
        # Subtract the start date from the end date
        ride_durations = rides['End date'] - rides['Start date']

        # Convert the results to seconds
        rides['Duration'] = ride_durations.dt.total_seconds()
        
        # Add a column for the weekday of the start of the ride
        # rides['Ride start weekday'] = rides['Start date'].dt.weekday_name
        rides['Ride start weekday'] = rides['Start date'].dt.day_name

        # Print the median trip time per weekday
        print(rides.groupby('Ride start weekday')['Duration'].median())

        # Shift the index of the end date up one; now subract it from the start date
        rides['Time since'] = rides['Start date'] - (rides['End date'].shift(1))

        # Move from a timedelta to a number of seconds, which is easier to work with
        rides['Time since'] = rides['Time since'].dt.total_seconds()

        # Resample to the month
        monthly = rides.resample('M', on = 'Start date')

        # Print the average hours between rides each month
        print(monthly['Time since'].mean()/(60*60))





def main():

    # chap3 = Timezones_Daylight()  
    # chap3.chap3_ex4(chap3.onebike_datetimes)

    # Import W20529's rides in Q4 2017
    import pandas as pd
    rides = pd.read_csv('capital-onebike.csv', parse_dates = ['Start date', 'End date'])
    # # Or: rides['Start date'] = pd.to_datetime(rides['Start date'], format = "%Y-%m-%d %H:%M:%S")

    chap4 = Timezones_Daylight()
    chap4.chap4_ex4(rides)
    


if __name__ == "__main__":
    main()


