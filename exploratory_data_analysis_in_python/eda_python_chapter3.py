''' Chapter 3: Relationships
Up until this point, you've only looked at one variable at a time. 
In this chapter, you'll explore relationships between variables two at a time, 
using scatter plots and other visualizations to extract insights from a new dataset obtained from 
the Behavioral Risk Factor Surveillance Survey (BRFSS). 
You'll also learn how to quantify those relationships using correlation and simple regression.
'''
# from eda_python_chapter2 import Distributions
# from FOLDER_NAME import FILENAME
# from FILENAME import CLASS_NAME FUNCTION_NAME
from numpy.lib.function_base import select
from eda_python_chapter2 import Distributions 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from empiricaldist import Pmf
import seaborn as sns
import tables 


class Relationships():
    ''' class to check the data distribution'''

    def __init__ (self):
        # initialize the Distribution object
        # self.chap2_dist = Distributions()
        # initialize dataset for chapter 3

        self.brfss = pd.read_hdf('brfss.hdf5','brfss')
        
        self.height = self.brfss['HTM4']
        self.weight = self.brfss['WTKG3']
        self.age = self.brfss['AGE']
        # access chapter 2 functions via self.dist
        self.dist = Distributions()
    
    def plt_label(self, plt_show ,x_label, y_label):
        if plt_show == 1:
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()
    
    def scatter_plot(self, x_data, y_data, x_label, y_label, alp, mark):
        plt.plot(x_data, y_data, 'o', alpha = alp, markersize = mark)
        self.plt_label(1, x_label, y_label)

    def explore_relationships(self):
        # check if we can access other class data
        self.dist.verbose("begin chapter 3: Relationships")
        self.dist.verbose("head of brfss dataset", self.brfss.head())
        
        # scatter plot
        plt.plot(self.height, self.weight, 'o')
        self.plt_label(1, 'height in cm', 'weight in kg')

        # transparency 
        # plt.plot(self.height, self.weight, 'o', alpha = 0.02)

        # marker size
        # plt.plot(self.height, self.weight, 'o', alpha = 0.02, markersize = 1)

        # jittering (adding random noise)
        height_jitter = self.height + np.random.normal(0,2, size = len(self.brfss))
        weight_jitter = self.weight + np.random.normal(0,2, size = len(self.brfss))
        # more jittering 
        plt.plot(height_jitter, weight_jitter, 'o', alpha = 0.01, markersize = 1)

        # Zoom in ( height from 140cm to 200 cm, and weight up to 160 kg)
        plt.axis([140, 200, 0, 160])

        # display
        self.plt_label(1, 'height in cm', 'weight in kg')

    def pmf_age(self):
        # Plot the PMF
        self.dist.pmf_plot(Pmf.from_seq(self.age, normalize = True),'AGE', 'Age in years','PMF')

    def scatter_plot_weight_age(self):
        # Select the first 1000 respondents
        brfss = self.brfss[:1000]

        # Extract age and weight
        age = brfss['AGE']
        weight = brfss['WTKG3']

        # Make a scatter plot
        self.scatter_plot(age, weight, 'Age in years', 'Weight in kg', 0.1, 1)

        # Add jittering to age
        age_jitter = age + np.random.normal(0, 0.5, size=len(brfss))
        self.scatter_plot(age_jitter, weight, 'Age in years', 'Weight in kg', 0.2, 5)

        # more jittering (more data)
        weight_jitter = weight + np.random.normal(0, 2, size=len(brfss))
        self.scatter_plot(age_jitter, weight_jitter, 'Age in years', 'Weight in kg', 0.2, 1)

    def violin_plot(self):
        data = self.brfss.dropna(subset = ['AGE', 'WTKG3'])
        sns.violinplot( x = 'AGE', y = 'WTKG3', data = data, inner = None)
        self.plt_label(1, 'Age in years', 'Weight in kg')

    def box_plot(self):
        data = self.brfss.dropna(subset = ['AGE', 'WTKG3'])
        # show pdf relations for each column of age
        sns.boxplot( x = 'AGE', y = 'WTKG3', data = data, whis = 10)
        self.plt_label(0, 'Age in years', 'Weight in kg')   

        # adding log scale for y
        plt.yscale('log')
        self.plt_label(1, 'Age in years', 'Weight in kg (log scale)') 

    def height_weight(self):
         # Drop rows with missing data
        data = self.brfss.dropna(subset=['_HTMG10', 'WTKG3'])
        # Make a box plot
        sns.boxplot(x='_HTMG10', y='WTKG3', data=data, whis=10)
        # Plot the y-axis on a log scale
        plt.yscale('log')

        # Remove unneeded lines and label axes
        sns.despine(left=True, bottom=True)
        plt.xlabel('Height in cm')
        plt.ylabel('Weight in kg')
        plt.show()

    def income_height(self):
        # Extract income
        income = self.brfss['INCOME2']
        # Plot the PMF
        Pmf(income).bar()
        # Label the axes
        plt.xlabel('Income level')
        plt.ylabel('PMF')
        plt.show()

        # Drop rows with missing data
        data = self.brfss.dropna(subset=['INCOME2', 'HTM4'])
        # Make a violin plot
        sns.violinplot( x = 'INCOME2', y = 'HTM4', data = data, inner = None)
        # Remove unneeded lines and label axes
        sns.despine(left=True, bottom=True)
        plt.xlabel('Income level')
        plt.ylabel('Height in cm')
        plt.show() 

    def test(self):
        # do stuff
        print('stuff')




def main():
    # main
    chap3 = Relationships()

    ## Exploring relationships
    # chap3.explore_relationships()

    ## PMF of Age
    # chap3.pmf_age()

    ## scatter plot of first 1000 rows for weight vs age, then add jittering
    # chap3.scatter_plot_weight_age()

    ## Violin plot to show pdf relations for each column
    # chap3.violin_plot()

    ## Box plot
    # chap3.box_plot()

    ## height and weight
    # chap3.height_weight()

    ## Income and height 
    # chap3.income_height()

    ## Correlation

    chap3.test()    


if __name__ == "__main__":
    main()


