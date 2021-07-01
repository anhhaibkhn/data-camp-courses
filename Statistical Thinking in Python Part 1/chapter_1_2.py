
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
        self.prnt(self.df.head(), HIDE)

        self.versicolor_petal_length = self.df.loc[self.df['target'] == 1., 'petal length (cm)']
        self.prnt(self.versicolor_petal_length.head(), HIDE)

        # adding the other species
        self.setosa_petal_length    = self.df.loc[self.df['target'] == 0., 'petal length (cm)']
        self.virginica_petal_length = self.df.loc[self.df['target'] == 2., 'petal length (cm)']

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
        # Empiricalcumulativedistributionfunction(ECDF)
        x = np.sort(df_swing['dem_share'])
        y = np.arange(1, len(x)+ 1) / len(x)

        _ = plt.plot(x,y, marker = '.', linestyle = 'none')
        _ = plt.ylabel('ECDF')
        _ = plt.xlabel('Percent of vote for Obama')
        plt.margins(0.02) # keeps data off plot edges

        plt.show()

    def ecdf(self, data):
        """Compute ECDF for a one-dimensional array of measurements."""
        # Number of data points: n
        n = len(data)

        # x-data for the ECDF: x
        x = np.sort(data)

        # y-data for the ECDF: y
        y = np.arange(1, n + 1) / n

        return x, y

    def plotting_ecdf(self, versicolor_petal_length):
        # Compute ECDF for versicolor data: x_vers, y_vers
        x_vers, y_vers = self.ecdf(versicolor_petal_length)

        # Generate plot
        _ = plt.plot(x_vers,y_vers, marker = '.', linestyle = 'none')

        # Label the axes
        _ = plt.ylabel('ECDF')
        _ = plt.xlabel('versicolor_petal_length')


        # Display the plot
        plt.show()

    def comparison_ECDFs(self, setosa_petal_length, versicolor_petal_length, virginica_petal_length):
        # Compute ECDFs
        x_set, y_set   = self.ecdf(setosa_petal_length)
        x_vers, y_vers = self.ecdf(versicolor_petal_length)
        x_virg, y_virg = self.ecdf(virginica_petal_length)

        # Plot all ECDFs on the same plot
        _ = plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
        _ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
        _ = plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')

        # Annotate the plot
        plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
        _ = plt.xlabel('petal length (cm)')
        _ = plt.ylabel('ECDF')

        # Display the plot
        plt.show()


""" Chap 2: Quantitative Exploratory Data Analysis

In this chapter, you will compute useful summary statistics, 
which serve to concisely describe salient features of a dataset with a few numbers."""


class Quantitative_EDA(Preparing_data):

    def __init__(self):
        super().__init__()

    def computing_mean(self, versicolor_petal_length):
        # Compute the mean: mean_length_vers
        mean_length_vers = np.mean(versicolor_petal_length)

        # Print the result with some nice formatting
        print('I. versicolor:', mean_length_vers, 'cm')

    def Computing_percentiles(self,versicolor_petal_length):
        # Specify array of percentiles: percentiles
        percentiles = np.array([2.5, 25, 50, 75, 97.5])

        # Compute percentiles: ptiles_vers
        ptiles_vers = np.percentile(versicolor_petal_length ,percentiles)

        # Print the result
        print(ptiles_vers) 
        
        x_vers, y_vers = self.ecdf(versicolor_petal_length)
        # Plot the ECDF
        _ = plt.plot(x_vers, y_vers, '.')
        _ = plt.xlabel('petal length (cm)')
        _ = plt.ylabel('ECDF')

        # Overlay percentiles as red diamonds.
        _ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
                linestyle='none')

        # Show the plot
        plt.show()

    def create_boxplot(self, df):
        # Create box plot with Seaborn's default settings
        _ = sns.boxplot(x='species', y='petal length (cm)', data=df)

        # Label the axes
        _ = plt.xlabel('species')
        _ = plt.ylabel('petal length (cm)')


        # Show the plot
        plt.show()

    
    def Computing_variance (self,versicolor_petal_length):
        # Array of differences to mean: differences
        differences = versicolor_petal_length - np.mean(versicolor_petal_length)

        # Square the differences: diff_sq
        diff_sq = differences ** 2

        # Compute the mean square difference: variance_explicit
        variance_explicit = np.mean(diff_sq)

        # Compute the variance using NumPy: variance_np
        variance_np = np.var(versicolor_petal_length)

        # Print the results
        print(variance_explicit, variance_np)

        # Compute the variance: variance
        variance = np.var(versicolor_petal_length)

        # Print the square root of the variance
        print(np.sqrt(variance))
        # Print the standard deviation
        print(np.std(versicolor_petal_length))


    def Making_scatter_plots(self, versicolor_petal_length, versicolor_petal_width):
        # Make a scatter plot
        _ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')

        # Label the axes
        _ = plt.xlabel('versicolor_petal_length')
        _ = plt.ylabel('versicolor_petal_width')

        # Show the result
        plt.show()

    """  Covariance and the Pearson correlation coefficient """
    def Computing_covariance_matrix(self,versicolor_petal_length, versicolor_petal_width):
        # Compute the covariance matrix: covariance_matrix
        covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

        # Print covariance matrix
        print(covariance_matrix)

        # Extract covariance of length and width of petals: petal_cov
        petal_cov = [covariance_matrix[0,1], covariance_matrix[1,0]]

        # Print the length/width covariance
        print(petal_cov)


    def Calculate_preason_r(self, versicolor_petal_length, versicolor_petal_width):
        def pearson_r(x, y):
            """Compute Pearson correlation coefficient between two arrays."""
            # Compute correlation matrix: corr_mat
            corr_mat = np.corrcoef(x, y)


            # Return entry [0,1]
            return corr_mat[0,1]

        # Compute Pearson correlation coefficient for I. versicolor: r
        r = pearson_r(versicolor_petal_length, versicolor_petal_width)

        # Print the result
        print(r)
        




def main():

    chap1 = Preparing_data()  

    # chap1.Plotting_Histogram(chap1.versicolor_petal_length)
    # chap1.Bee_swarm_plot(chap1.df_swing)
    # chap1.making_ECDF(chap1.df_swing)
    # chap1.plotting_ecdf(chap1.versicolor_petal_length)
    # chap1.comparison_ECDFs(chap1.setosa_petal_length,\
    #                         chap1.versicolor_petal_length,\
    #                         chap1.virginica_petal_length)

    chap2 = Quantitative_EDA()
    # chap2.Computing_percentiles(chap2.versicolor_petal_length)
    chap2.create_boxplot(chap2.df)


if __name__ == "__main__":
    main()