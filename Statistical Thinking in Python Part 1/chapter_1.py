
""" Statistical Thinking in Python (Part 1)

Course Description:

After all of the hard work of acquiring data and getting them into a form you can work with, 
you ultimately want to make clear, succinct conclusions from them. This crucial last step of 
a data analysis pipeline hinges on the principles of statistical inference. 
In this course, you will start building the foundation you need to think statistically, 
speak the language of your data, and understand what your data is telling you. 
The foundations of statistical thinking took decades to build, but can be grasped much faster today 
with the help of computers. With the power of Python-based tools, you will rapidly get up-to-speed 
 and begin thinking statistically by the end of this course.
 
 """
""" Chapter 1: Graphical Exploratory Data Analysis

Before diving into sophisticated statistical inference techniques, you should first explore your data 
by plotting them and computing simple summary statistics. This process, called exploratory data analysis, 
is a crucial first step in statistical analysis of data.

"""

# Import plotting modules
import matplotlib.pyplot as plt 
import seaborn as sns   
from sklearn.datasets import load_iris

import numpy as np
import pandas as pd

SHOW = True
HIDE = False

class Preparing_data():

    def __init__(self):
        # save load_iris() sklearn dataset to iris
        # if you'd like to check dataset type use: type(load_iris())
        # if you'd like to view list of attributes use: dir(load_iris())
        iris = load_iris()

        self.df = pd.DataFrame(data= np.c_[iris['data'], iris['target'],],
                     columns= iris['feature_names'] + ['target'] )

        # iris.target = [0 0 0 ... 1 2]
        # iris.target_name = ['setosa' 'versicolor' 'virginica']
        self.df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

        # print(iris.target, iris.target_names)
        self.prnt(self.df.head())

        self.versicolor_petal_length = self.df.loc[self.df['target'] == 1., 'petal length (cm)']
        self.prnt(self.versicolor_petal_length.head(), HIDE)

        # loeading df swing
        self.df_swing = pd.read_csv('2008_swing_states.csv')


    

    def prnt(self, data, display = SHOW):
        if display:
            print('###############', '\n', data, '\n')


    def Plotting_Histogram(self, versicolor_petal_length):
        # Set default Seaborn style
        sns.set()

        # Plot histogram of versicolor petal lengths
        _ = plt.hist(versicolor_petal_length)

        # Label axes
        _ = plt.xlabel('petal length (cm)')
        _ = plt.ylabel('count')

        # Show histogram
        plt.show()
    
    def Hist_adjusting_bins(self, versicolor_petal_length):
        # Compute number of data points: n_data
        n_data = len(versicolor_petal_length)

        # Number of bins is the square root of number of data points: n_bins
        n_bins = np.sqrt(n_data)

        # Convert number of bins to integer: n_bins
        n_bins = int(n_bins)

        # Plot the histogram
        _ = plt.hist(versicolor_petal_length, bins = n_bins)

        # Label axes
        _ = plt.xlabel('petal length (cm)')
        _ = plt.ylabel('count')

        # Show histogram
        plt.show()

    def Bee_swarm_plot(self, df_swing):

        self.prnt(df_swing.head())

        _ = sns.swarmplot(x = 'state', y = 'dem_share', data = df_swing)

        # Label axes
        _ = plt.xlabel('State')
        _ = plt.ylabel('Percent of vote for Obama')
        
        plt.show()

    def Bee_swarm_plot2(self, df):

        # Create bee swarm plot with Seaborn's default settings
        _ = sns.swarmplot(x='species', y='petal length (cm)', data=df)
        _ = plt.xlabel('species')
        _ = plt.ylabel('petal length (cm)')
        plt.show()

    def making_ECDF(self, df_swing):

        x = np.sort(df_swing['dem_share'])
        y = np.arange(1, len(x)+ 1) / len(x)

        _ = plt.plot(x,y, marker = '.', linestyle = 'none')
        _ = plt.ylabel('ECDF')
        _ = plt.xlabel('Percent of vote for Obama')
        plt.margins(0.02) # keeps data off plot edges

        plt.show()




def main():

    chap1 = Preparing_data()  

    # chap1.Plotting_Histogram(chap1.versicolor_petal_length)
    # chap1.Bee_swarm_plot(chap1.df_swing)
    chap1.making_ECDF(chap1.df_swing)




if __name__ == "__main__":
    main()