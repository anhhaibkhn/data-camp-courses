""" Chapter 1:
Introduction to Seaborn

What is Seaborn, and when should you use it? In this chapter, you will find out! 
Plus, you will learn how to create scatter plots and count plots with both lists of data 
and pandas DataFrames. You will also be introduced to one of the big advantages of 
using Seaborn - the ability to easily add a third variable to your plots by using color 
to represent different subgroups.

"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Seaborn_Intermediate():

    def __init__(self):
        self.bicycle_df = pd.read_csv("WashingtonDCBikeShare_bike_share.csv")
        # self.summary_df(self.bicycle_df )
        self.college_df  = pd.read_csv("college_datav3.csv")

    def summary_df(self,df):
        print("Summary of the basic information about this DataFrame and its data:")
        print(df.info())
        print("Top 5 rows:")
        print(df.head())
        print("Statistical data:")
        print(df.describe())


    def exercise_1(self):
        """Regression plot"""
        sns.regplot(data=self.bicycle_df, x='temp', y ='total_rentals', marker='+')
        plt.title('regression plot')
        plt.show()

        sns.residplot(data=self.bicycle_df, x='temp', y ='total_rentals')
        plt.title('residual plot')
        plt.show()

        sns.regplot(data=self.bicycle_df, x='temp', y ='total_rentals', order=2)
        plt.title('regression plot with polynomial = 2')
        plt.show()

        sns.regplot(data=self.bicycle_df, x='mnth', y ='total_rentals',
                                                    x_jitter=.1, order=2)
        plt.title('regression plot with x_jitter folliwng months')
        plt.show()

        sns.regplot(data=self.bicycle_df, x='mnth', y ='total_rentals',
                                                    x_estimator = np.mean, order=2)
        plt.title('regression plot with x_estimator = np.mean, folliwng months')
        plt.show()

        sns.regplot(data=self.bicycle_df, x='temp', y ='total_rentals',
                                                    x_bins= 12)
        plt.title('regression plot with Binning the data')
        plt.show()

    def exercise_2(self):
        # Display a regression plot for Tuition
        sns.regplot(data=self.college_df, y='Tuition',
                    x="SAT_AVG_ALL",
                    marker='^',
                    color='g')

        plt.title('regression plot for Tuition')
        plt.show()
        plt.clf()

        # Display the residual plot
        sns.residplot(data=self.college_df,
                y='Tuition',
                x="SAT_AVG_ALL",
                color='g')

        plt.title('residual plot for Tuition')
        plt.show()
        plt.clf()

    def exercise_3(self):
        # Plot a regression plot of Tuition and the Percentage of Pell Grants
        sns.regplot(data=self.college_df,
                    y='Tuition',
                    x="PCTPELL")
        plt.title('Plot a regression plot of Tuition and the Percentage of Pell Grants')
        plt.show()
        plt.clf()


def main():
    chap3 = Seaborn_Intermediate()
    # chap3.exercise_1()
    # chap3.exercise_2()
    chap3.exercise_3()


if __name__ == "__main__":
    main()