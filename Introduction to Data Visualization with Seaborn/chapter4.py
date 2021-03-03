""" Chapter 4:
Customizing Seaborn Plots

In this final chapter, you will learn how to add informative plot titles and axis labels, 
which are one of the most important parts of any data visualization! You will also learn 
how to customize the style of your visualizations in order to more quickly orient your audience 
to the key takeaways. Then, you will put everything you have learned together for 
the final exercises of the course!

"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from chapter1 import Seaborn_Intro
from chapter2 import Seaborn_Visualize_two_vars
from chapter3 import Seaborn_Categorical
import numpy as np


class Seaborn_Customization():

    def __init__(self):
        self.chap1 = Seaborn_Intro()
        self.chap2 = Seaborn_Visualize_two_vars()
        self.chap3 = Seaborn_Categorical()

        self.survey_data = self.chap3.survey_data
        ### ADD Parents Advice
        # Drop Parents' advice NaN
        self.survey_data = self.survey_data[self.survey_data["Parents' advice"].notna()]
        # create a list of our conditions
        conditions = [
            (self.survey_data["Parents' advice"] == 1.0),
            (self.survey_data["Parents' advice"] == 2.0),
            (self.survey_data["Parents' advice"] == 3.0),
            (self.survey_data["Parents' advice"] == 4.0),
            (self.survey_data["Parents' advice"] == 5.0)]

        # create a list of the values we want to assign for each condition
        values = ["Never", "Rarely", "Sometimes","Often", "Always"]

        # create a new column and use np.select to assign values to it using our lists as arguments
        self.survey_data["Parents Advice"] = np.select(conditions, values)

        ### ADD Feels Lonely
        # Drop Loneliness NaN
        self.survey_data = self.survey_data[self.survey_data["Loneliness"].notna()]
        # create a list of our conditions
        conditions_2 = [
            (self.survey_data["Loneliness"] <= 3.0),
            (self.survey_data["Loneliness"] > 3.0)]

        # create a list of the values we want to assign for each condition
        values_2 = [False, True]

        # create a new column and use np.select to assign values to it using our lists as arguments
        self.survey_data["Feels Lonely"] = np.select(conditions_2, values_2)

        print(self.survey_data.head())
        print(self.survey_data['Loneliness'].describe())
        print(self.survey_data['Loneliness'].unique())

        ### ADD "Interested in Pets"
        # Drop Pets NaN
        self.survey_data = self.survey_data[self.survey_data["Pets"].notna()]
        # create a list of our conditions
        conditions_3 = [
            (self.survey_data["Pets"] > 3.0),
            (self.survey_data["Pets"] <= 3.0)]

        # create a list of the values we want to assign for each condition
        values_3 = ["Yes", "No"]

        # create a new column and use np.select to assign values to it using our lists as arguments
        self.survey_data["Interested in Pets"] = np.select(conditions_3, values_3)

        ### ADD "Likes Techno"
        # Drop Pets NaN
        self.survey_data = self.survey_data[self.survey_data["Techno"].notna()]
        # create a list of our conditions
        conditions_4 = [
            (self.survey_data["Techno"] < 5.0),
            (self.survey_data["Techno"] >= 5.0)]

        # create a list of the values we want to assign for each condition
        values_4 = [False, True]

        # create a new column and use np.select to assign values to it using our lists as arguments
        self.survey_data["Likes Techno"] = np.select(conditions_4, values_4)


    def exercise_1(self):
        """ Changing style and palette  
         How often do you listen to your parents' advice? """
        survey_data = self.survey_data
        print(survey_data.describe())
        print(survey_data.info())
        print(survey_data["Parents Advice"].unique())

        # Set the style to "whitegrid"
        # Change the color palette to "RdBu"
        sns.set_style("whitegrid")
        sns.set_palette("RdBu")
        # Set the color palette to "Purples"
        # sns.set_palette("Purples")

        # Create a count plot of survey responses
        category_order = ["Never", "Rarely", "Sometimes", 
                        "Often", "Always"]

        sns.catplot(x="Parents Advice", 
                    data=survey_data, 
                    kind="count", 
                    order=category_order)

        # Show plot
        plt.show()

    def exercise_2(self):
        """ Changing the scale """
         ### ADD Number of Siblings
        # Drop Siblings NaN
        self.survey_data = self.survey_data[self.survey_data["Siblings"].notna()]
        # create a list of our conditions
        conditions = [
            (self.survey_data["Siblings"] == 0.0),
            (self.survey_data["Siblings"] > 1.0)&(self.survey_data["Siblings"] < 3.0),
            (self.survey_data["Siblings"] >= 3.0)]

        # create a list of the values we want to assign for each condition
        values = ["0", "1 - 2", "3+"]

        # create a new column and use np.select to assign values to it using our lists as arguments
        self.survey_data["Number of Siblings"] = np.select(conditions, values)

        survey_data = self.survey_data

        # Change the context to "notebook"
        # Smallest to largest:"paper","notebook","talk","poster"
        sns.set_context("poster")

        # Create bar plot
        sns.catplot(x="Number of Siblings", y="Feels Lonely",
                    data=survey_data, kind="bar")

        # Show plot
        plt.show()

        

    def exercise_3(self):
        """ Using a custom palette 
        a basic summary of the type of people answering this survey
        howing the distribution of ages for male versus female respondents"""
        survey_data = self.survey_data

        # Set the style to "darkgrid"
        sns.set_style("darkgrid")

        # Set a custom color palette
        custom_palette = ['#39A7D0',"#36ADA4"]
        sns.set_palette(custom_palette)

        # Create the box plot of age distribution by gender
        sns.catplot(x="Gender", y="Age", 
                    data=survey_data, kind="box")

        # Show plot
        plt.show()

        
    
    def exercise_4_1(self):
        """ Adding titles and labels 
        FacetGrids vs. AxesSubplots """
        mpg = self.chap2.mpg

        # Create scatter plot
        g = sns.relplot(x="weight", 
                        y="horsepower", 
                        data=mpg,
                        kind="scatter")

        # Add a title "Car Weight vs. Horsepower"
        g.fig.suptitle("Car Weight vs. Horsepower")

        # Show plot
        plt.show()

    def exercise_4_2(self, mpg_mean):
        """ Adding titles and labels 2
        Adding a title and axis labels """
        
        # Create line plot
        g = sns.lineplot(x="model_year", y="mpg_mean", 
                        data=mpg_mean,
                        hue="origin")

        # Add a title "Average MPG Over Time"
        g.set_title("Average MPG Over Time")

        # Add x-axis and y-axis labels
        g.set(xlabel = "Car Model Year",
            ylabel = "Average MPG")

        # Show plot
        plt.show()

    
    def exercise_4_2(self):
        """ Adding titles and labels 3
        Rotating x-tick labels """
        mpg = self.chap2.mpg

        # Create point plot
        sns.catplot(x="origin", 
                    y="acceleration", 
                    data=mpg, 
                    kind="point", 
                    join=False, 
                    capsize=0.1)

        # Rotate x-tick labels
        plt.xticks(rotation = 90)

        # Show plot
        plt.show()

    def exercise_5_1(self):
        """ Box plot with subgroupss """
        survey_data = self.survey_data
        # Set palette to "Blues"
        sns.set_palette("Blues")

        # Adjust to add subgroups based on "Interested in Pets"
        g = sns.catplot(x="Gender",
                        y="Age", data=survey_data, 
                        kind="box", hue="Interested in Pets")

        # Set title to "Age of Those Interested in Pets vs. Not"
        g.fig.suptitle("Age of Those Interested in Pets vs. Not")

        # Show plot
        plt.show()

    def exercise_5_2(self):
        """ Bar plot with subgroups and subplots """
        survey_data = self.survey_data
        # Set the figure style to "dark"
        sns.set_style("dark")

        # Adjust to add subplots per gender
        g = sns.catplot(x="Village - town", y="Likes Techno", 
                        data=survey_data, kind="bar",
                        col= "Gender")

        # Add title and axis labels
        g.fig.suptitle("Percentage of Young People Who Like Techno", y=1.02)
        g.set(xlabel="Location of Residence", 
            ylabel="% Who Like Techno")

        # Show plot
        plt.show()


def main():
    chap3 = Seaborn_Customization()
    # chap3.exercise_1()
    # chap3.exercise_2()
    # chap3.exercise_3()
    # chap3.exercise_4_1()
    # chap3.exercise_4_2()
    # chap3.exercise_4_3()
    # chap3.exercise_5_1()
    chap3.exercise_5_2()


if __name__ == "__main__":
    main()