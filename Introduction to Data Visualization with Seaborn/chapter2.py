""" Chapter 2:
Visualizing Two Quantitative Variables

In this chapter, you will create and customize plots that visualize 
the relationship between two quantitative variables. 
To do this, you will use scatter plots and line plots to explore 
how the level of air pollution in a city changes over the course of a day 
and how horsepower relates to fuel efficiency in cars. 
You will also see another big advantage of using Seaborn - 
the ability to easily create subplots in a single figure!

"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from chapter1 import Seaborn_Intro


class Seaborn_Visualize_two_vars():

    def __init__(self):
        self.chap1 = Seaborn_Intro()
        self.mpg = pd.read_csv("mpg.csv")

    def exercise_1(self):
        """ Creating subplots with col and row """
        student_data = self.chap1.student_data
        print(student_data.head())

        # Change to use relplot() instead of scatterplot()
        ## Change to make subplots based on study time
        # Change this scatter plot to arrange the plots in rows instead of columns
        sns.relplot(x="absences", y="G3", 
                    data=student_data,
                    kind="scatter", 
                    row="study_time")

        # Show plot
        plt.show()

    def exercise_2(self):
        """ Creating two-factor subplots """
        student_data = self.chap1.student_data
        # Adjust to add subplots based on school support
        # Adjust further to add subplots based on family support
        sns.relplot(x="G1", y="G3", 
                    data=student_data,
                    kind="scatter", 
                    col="schoolsup",
                    col_order=["yes", "no"],
                    row="famsup",
                    row_order=["yes", "no"])

        # Show plot
        plt.show()

    def exercise_3_1(self):
        """ customizing scatter plots 
            Changing the size of scatter plot points """

        mpg = self.mpg

        # Create scatter plot of horsepower vs. mpg
        sns.relplot(x="horsepower", y="mpg", 
                    data=mpg, kind="scatter", 
                    size="cylinders", 
                    hue="cylinders")
        # Show plot
        plt.show()

    def exercise_3_2(self):
        """ customizing scatter plots 
           Changing the style of scatter plot points """

        mpg = self.mpg

        # Create a scatter plot of acceleration vs. mpg
        # Vary the style and color of the plot points by country of origin
        sns.relplot(x="acceleration", y="mpg", 
                            data=mpg, kind="scatter", 
                            style="origin", 
                            hue="origin")
        # Show plot
        plt.show()

    def exercise_4(self):
        """" Line plots for time series
              Interpreting line plots  """
        mpg = self.mpg    
        # Make the shaded area show the standard deviation
        sns.relplot(x="model_year", y="mpg",
                    data=mpg, kind="line", ci = "sd")

        # Show plot
        plt.show()
    
    def exercise_5(self):
        """ Line plots for time series
            Plotting subgroups in line plots """
        mpg = self.mpg   
        # Add markers and make each line have the same style
        sns.relplot(x="model_year", y="horsepower", 
                    data=mpg, kind="line", 
                    ci=None, style="origin", 
                    hue="origin", markers=True,
                    dashes=False)

        # Show plot
        plt.show()




def main():
    chap2 = Seaborn_Visualize_two_vars()
    # chap2.exercise_1()
    # chap2.exercise_2()
    chap2.exercise_3_2()
    # chap2.exercise_4()
    # chap2.exercise_5()


if __name__ == "__main__":
    main()