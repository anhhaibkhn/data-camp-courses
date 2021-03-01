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

class Seaborn_Intro():

    def __init__(self):
        self.countries_df = pd.read_csv("countries-of-the-world.csv")
        self.columns = list(self.countries_df.columns.values)
        # print(self.countries_df.info())
        # print(self.columns)
        self.gdp = self.countries_df['GDP ($ per capita)'].values.tolist()
        # convert object to string then float
        self.phones = self.countries_df['Phones (per 1000)'].apply(lambda x: float(str(x).replace(',','.'))).values.tolist()
        self.percent_literate = self.countries_df['Literacy (%)'].apply(lambda x: float(str(x).replace(',','.'))).values.tolist()

        #student dataset
        self.student_data = pd.read_csv("student-alcohol-consumption.csv")


    def exercise_1(self):
        """ Making a scatter plot with lists """
        gdp = self.gdp
        phones = self.phones 
        percent_literate =  self.percent_literate
        # print(len(gdp), len(phones),len(percent_literate))
        print(type(self.percent_literate[1]))
        print((percent_literate[1]))

        # Create scatter plot with GDP on the x-axis and number of phones on the y-axis
        sns.scatterplot(x = gdp, y = phones)
        plt.show()

        # Change this scatter plot to have percent literate on the y-axis
        # sns.scatterplot(x=gdp, y=percent_literate)      
        # plt.show()

    def exercise_2(self):
        """ Making a count plot with a list """
        region = self.countries_df["Region"]
        ylabels = [s.rstrip() for s in region.unique().tolist() ]
        print(ylabels)

        # Create count plot with region on the y-axis
        ax = sns.countplot(y=region)
        ax.set_yticklabels(ylabels, rotation=0, fontsize="9", va="center")

        # Show plot
        plt.show()

    def exercise_3(self):
        """ Making a count plot with a DataFrame """
        csv_filepath = "young-people-survey-responses.csv"

        # Create a DataFrame from csv file
        df = pd.read_csv(csv_filepath)
        print(df["Spiders"].head())

        # Create a count plot with "Spiders" on the x-axis
        sns.countplot(x = "Spiders", data = df)

        # Display the plot
        plt.show()

    def exercise_4(self):
        """ Hue and scatter plots """
        student_data = self.student_data
        # Change the legend order in the scatter plot
        sns.scatterplot(x="absences", y="G3", 
                        data=student_data, 
                        hue="location",
                        hue_order = ["Rural"
                                    ,"Urban"])

        # Show plot
        plt.show()

    def exercise_5(self):
        """ Hue and scatter plots """
        student_data = self.student_data
        # Create a dictionary mapping subgroup values to colors
        palette_colors = {"Rural": "green", "Urban": "blue"}

        # Create a count plot of school with location subgroups
        sns.countplot(x="school", data=student_data
                                , hue = "location"
                                , palette = palette_colors)


        # Display plot
        plt.show()



def main():
    chap1 = Seaborn_Intro()
    # chap1.exercise_1()
    # chap1.exercise_2()
    # chap1.exercise_3()
    # chap1.exercise_4()
    chap1.exercise_5()


if __name__ == "__main__":
    main()