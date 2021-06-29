""" Chapter 3: Visual exploratory data analysis

Are you more likely to get arrested at a certain time of day? Are drug-related stops on the rise? 
In this chapter, you will answer these and other questions by analyzing the dataset visually, 
since plots can help you to understand trends in a way that examining the raw data cannot. """

from chapter1_2 import Preparing_data
import matplotlib.pyplot as plt
import pandas as pd


class Visual_EDA(Preparing_data):

    def __init__(self):
        super().__init__()

        self.ri = self.exercise_1(self.ri)


    def ex1(self, ri):
        # Calculate the overall arrest rate
        print(ri.is_arrested.mean())

        # Calculate the hourly arrest rate
        print(ri.groupby(ri.index.hour).is_arrested.mean())

        # Save the hourly arrest rate
        hourly_arrest_rate = ri.groupby(ri.index.hour).is_arrested.mean()

        # Create a line plot of 'hourly_arrest_rate'
        plt.plot(hourly_arrest_rate)

        # Add the xlabel, ylabel, and title
        plt.xlabel('Hour')
        plt.ylabel('Arrest Rate')
        plt.title('Arrest Rate by Time of Day')

        # Display the plot
        """ The arrest rate has a significant spike overnight, 
             and then dips in the early morning hours. """

        plt.show()
    
    def ex2(self, ri):
        
        # Calculate the annual rate of drug-related stops
        print(ri.drugs_related_stop.resample('A').mean())

        # Save the annual rate of drug-related stops
        annual_drug_rate = ri.drugs_related_stop.resample('A').mean()

        # Create a line plot of 'annual_drug_rate'
        plt.plot(annual_drug_rate)

        # Display the plot
        plt.show()
    
    def ex3(self, ri):
        # Save the annual rate of drug-related stops
        annual_drug_rate = ri.drugs_related_stop.resample('A').mean()
        # Calculate and save the annual search rate
        annual_search_rate = ri.search_conducted.resample('A').mean()

        # Concatenate 'annual_drug_rate' and 'annual_search_rate'
        annual = pd.concat([annual_drug_rate,annual_search_rate], axis='columns')

        # Create subplots from 'annual'
        annual.plot(subplots = True)

        # Display the subplots
        plt.show()

    def ex4(self, ri):
        """ What violations are caught in each district """
        # create a cross tabulation with crosstab with 2 pd series
        print("# computing a frequency table: ")
        table = pd.crosstab(ri.driver_race, ri.driver_gender)
        print(table.head())

        asian_female = ri[(ri.driver_race == 'Asian') & (ri.driver_gender == 'F')]
        print(asian_female.shape)

        A_H = table.loc['Asian':'Hispanic']
        print(table.loc['Asian':'Hispanic'])
        A_H.plot()
        plt.show()




def main():
    chap3 = Visual_EDA()
    chap3.ex4(chap3.ri)



if __name__ == "__main__":
    main()

    
