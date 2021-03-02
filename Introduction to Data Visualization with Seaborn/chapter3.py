""" Chapter 3:
Visualizing a Categorical and a Quantitative Variable

Categorical variables are present in nearly every dataset, but they are especially prominent in survey data. 
In this chapter, you will learn how to create and customize categorical plots such as 
box plots, bar plots, count plots, and point plots. Along the way, you will explore survey data from young people 
about their interests, students about their study habits, and adult men about their feelings about masculinity.

"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from chapter1 import Seaborn_Intro
from chapter2 import Seaborn_Visualize_two_vars
import numpy as np


class Seaborn_Categorical():

    def __init__(self):
        self.chap1 = Seaborn_Intro()
        self.chap2 = Seaborn_Visualize_two_vars()
        self.survey_data = pd.read_csv("young-people-survey-responses.csv")

        # Drop Age NaN
        self.survey_data = self.survey_data[self.survey_data['Age'].notna()]
        # create a list of our conditions
        conditions = [
            (self.survey_data['Age'] < 21.0),
            (self.survey_data['Age'] >= 21.0)]

        # create a list of the values we want to assign for each condition
        values = ['Less than 21', '21+']

        # create a new column and use np.select to assign values to it using our lists as arguments
        self.survey_data['Age Category'] = np.select(conditions, values)

        print(self.survey_data.head())
        print(self.survey_data['Age Category'].describe())
        print(self.survey_data['Age Category'].unique())
        print(self.survey_data.loc[self.survey_data['Age Category'] == '0'])



    def exercise_1(self):
        """ Count plots We might suspect that young people spend 
        a lot of time on the internet  """
        survey_data = self.survey_data
        # Create count plot of internet usage
        # sns.catplot(x="Internet usage", data=survey_data, kind='count')
        # Create column subplots based on age category
        sns.catplot(col="Age Category",y="Internet usage", data=survey_data,
                    kind="count")

        # Show plot
        plt.show()

    def exercise_2(self):
        """ Bar plots with percentages """
        survey_data = self.survey_data

        # Drop Age NaN
        survey_data = survey_data[survey_data['Mathematics'].notna()]
        # create a list of our conditions
        conditions = [
            (survey_data['Mathematics'] < 4.0),
            (survey_data['Mathematics'] >= 4.0)]
        # create a list of the values we want to assign for each condition
        values = [False, True]

        # create a new column and use np.select to assign values to it using our lists as arguments
        survey_data['Interested in Math'] = np.select(conditions, values)
        print(survey_data.head())

        # Create a bar plot of interest in math, separated by gender
        sns.catplot(x="Gender" , y= "Interested in Math", data= survey_data, kind="bar")

        # Show plot
        plt.show()

    def exercise_3(self):
        """ Customizing bar plots """
        student_data = self.chap1.student_data
        # Create bar plot of average final grade in each study category
        # # Rearrange the categories in order
        # Turn off the confidence intervals
        sns.catplot(x="study_time", y="G3",
                    data=student_data,
                    kind="bar",
                    order=["<2 hours", 
                        "2 to 5 hours", 
                        "5 to 10 hours", 
                        ">10 hours"],
                    ci=None)

        # Show plot
        plt.show()
    
    def exercise_3(self):
        """ Create Box plots """
        student_data = self.chap1.student_data
        # Create bar plot of average final grade in each study category
        # # Rearrange the categories in order
        # Turn off the confidence intervals
        sns.catplot(x="study_time", y="G3",
                    data=student_data,
                    kind="bar",
                    order=["<2 hours", 
                        "2 to 5 hours", 
                        "5 to 10 hours", 
                        ">10 hours"],
                    ci=None)

        # Show plot
        plt.show()
    




def main():
    chap2 = Seaborn_Categorical()
    # chap2.exercise_1()
    # chap2.exercise_2()
    chap2.exercise_3()
    # chap2.exercise_4()
    # chap2.exercise_5()


if __name__ == "__main__":
    main()