""" COURSE: Statistical Thinking in Python (Part 2)

Course Description: 

After completing Statistical Thinking in Python (Part 1), you have the probabilistic mindset and 
foundational hacker stats skills to dive into data sets and extract useful information from them. 
In this course, you will do just that, expanding and honing your hacker stats toolbox to perform 
the two key tasks in statistical inference, parameter estimation and hypothesis testing. 
You will work with real data sets as you learn, culminating with analysis of measurements of the beaks 
of the Darwin's famous finches. You will emerge from this course with new knowledge and lots of practice 
under your belt, ready to attack your own inference problems out in the world.  """

""" Chapter 1: Parameter estimation by optimization 

When doing statistical inference, we speak the language of probability. 
A probability distribution that describes your data has parameters. 
So, a major goal of statistical inference is to estimate the values of these parameters, 
which allows us to concisely and unambiguously describe our data and draw conclusions 
from it. In this chapter, you will learn how to find the optimal parameters, 
those that best describe your data.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, 'E:/data_science_resources/git_data_camp/data-camp-courses/Statistical Thinking in Python Part 1/')
from chapter_1_2_3_4 import Preparing_data as prep


class Parameter:
    def __init__(self):
        
        self.anscombe_data = pd.read_csv("E:/data_science_resources/git_data_camp/data-camp-courses/Statistical Thinking in Python Part 2/anscombe.csv")
        print(self.anscombe_data.info(verbose=True))
        # get x, y data
        self.x, self.y = self.anscombe_data.iloc[1:,0].to_numpy().astype(np.float64), self.anscombe_data.iloc[1:,1].to_numpy().astype(np.float64)
        
        self.anscombe_x, self.anscombe_y = [], []
        # change the column name to be the first row 
        self.anscombe_data.columns = self.anscombe_data.iloc[0]

        # test with df name
        df = self.anscombe_data
        print(self.anscombe_data.info(verbose=True))

        for i in range(0,len(df.columns)):
            if i%2:
                self.anscombe_y.append(df.iloc[1:,i].to_numpy().astype(np.float64))  
            else: 
                self.anscombe_x.append(df.iloc[1:,i].to_numpy().astype(np.float64))

        


    def exponential_distribution(self, nohitter_times):
        # Seed random number generator
        np.random.seed(42)

        # Compute mean no-hitter time: tau
        tau = np.mean(nohitter_times)

        # Draw out of an exponential distribution with parameter tau: inter_nohitter_time
        inter_nohitter_time = np.random.exponential(tau, 100000)

        # Plot the PDF and label axes
        _ = plt.hist(inter_nohitter_time,
                    bins=50, normed=True, histtype='step')
        _ = plt.xlabel('Games between no-hitters')
        _ = plt.ylabel('PDF')

        # Show the plot
        plt.show()

    def compare_cdf(self, nohitter_times, inter_nohitter_time):
        # Create an ECDF from real data: x, y
        x, y = prep.ecdf(nohitter_times)

        # Create a CDF from theoretical samples: x_theor, y_theor
        x_theor, y_theor = prep.ecdf(inter_nohitter_time)

        # Overlay the plots
        plt.plot(x_theor, y_theor)
        plt.plot(x, y, marker='.', linestyle='none')

        # Margins and axis labels
        plt.margins(0.02)
        plt.xlabel('Games between no-hitters')
        plt.ylabel('CDF')

        # Show the plot
        plt.show()

    def test_ecdf_plot(self, sample):
        x, y = prep.ecdf(sample)

        _ = plt.plot(x, y  , marker = '.', linestyle = 'none')
        plt.show()
    
    def plot_cdfs(self, x_theor, y_theor, x, y, tau):
        # Plot the theoretical CDFs
        plt.plot(x_theor, y_theor)
        plt.plot(x, y, marker='.', linestyle='none')
        plt.margins(0.02)
        plt.xlabel('Games between no-hitters')
        plt.ylabel('CDF')

        # Take samples with half tau: samples_half
        samples_half = np.random.exponential(tau/2, size = 10000)

        # Take samples with double tau: samples_double
        samples_double = np.random.exponential(tau*2, size = 10000)

        # Generate CDFs from these samples
        x_half, y_half = prep.ecdf(samples_half)
        x_double, y_double = prep.ecdf(samples_double)

        # Plot these CDFs as lines
        _ = plt.plot(x_half, y_half)
        _ = plt.plot(x_double, y_double)

        # Show the plot
        plt.show()

    def pearson_r(self, x, y):
        """Compute Pearson correlation coefficient between two arrays."""
        # Compute correlation matrix: corr_mat
        corr_mat = np.corrcoef(x, y)

        # Return entry [0,1]
        return corr_mat[0,1]

    def correlation_plot(self, x, y ):
        illiteracy, fertility = x, y
        # Plot the illiteracy rate versus fertility
        _ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

        # Set the margins and label axes
        plt.margins(0.02)
        _ = plt.xlabel('percent illiterate')
        _ = plt.ylabel('fertility')

        # Show the plot
        plt.show()

        # Show the Pearson correlation coefficient
        print(self.pearson_r(illiteracy, fertility))


    def Linear_regression(self, illiteracy, fertility):
        """assume that fertility is a linear function of the female illiteracy rate. 
        That is, , where  is the slope and  is the intercept."""

        # Plot the illiteracy rate versus fertility
        _ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
        plt.margins(0.02)
        _ = plt.xlabel('percent illiterate')
        _ = plt.ylabel('fertility')

        # Perform a linear regression using np.polyfit(): a, b
        a, b = np.polyfit(illiteracy, fertility, 1)

        # Print the results to the screen
        print('slope =', a, 'children per woman / percent illiterate')
        print('intercept =', b, 'children per woman')

        # Make theoretical line to plot
        x = np.array([0,100])
        y = a * x + b

        # Add regression line to your plot
        _ = plt.plot(x, y)

        # Draw the plot
        plt.show()      
            

    def Optimall_with_RSS(self, fertility, illiteracy):  
        # Specify slopes to consider: a_vals
        a_vals = np.linspace(0,0.1,200)

        # Initialize sum of square of residuals: rss
        rss = np.empty_like(a_vals)

        # Compute sum of square of residuals for each value of a_vals
        for i, a in enumerate(a_vals):
            rss[i] = np.sum(( fertility - a* illiteracy - b)**2)

        # Plot the RSS
        plt.plot(a_vals,rss, '-')
        plt.xlabel('slope (children per woman / percent illiterate)')
        plt.ylabel('sum of square of residuals')

        plt.show()

    def linear_regression(self, x,y):
        """ Linear regression on appropriate Anscombe data """
        print(type(x), type(y), len(x), len(y))
        for dum in x:
            if type(dum) == str:
                print(dum) 
        # Perform linear regression: a, b
        a, b = np.polyfit(x, y, 1)

        # Print the slope and intercept
        print(a, b)

        # Generate theoretical x and y data: x_theor, y_theor
        x_theor = np.array([3, 15])
        y_theor = x_theor * a + b

        # Plot the Anscombe data and theoretical line
        _ = plt.plot(x, y, marker='.', linestyle='none')
        _ = plt.plot(x_theor, y_theor)

        # Label the axes
        plt.xlabel('x')
        plt.ylabel('y')
        
        # add grid
        plt.grid(color='white', linestyle='-', linewidth=0.8)
        ax = plt.axes()
        # Setting the background color of the
        # plot using set_facecolor() method
        ax.set_facecolor("lightsteelblue")

        # Show the plot
        plt.show()

    def linear_reg_on_Anscombe(self,anscombe_x , anscombe_y):
        # Iterate through x,y pairs
        for x, y in zip(anscombe_x , anscombe_y ):
            # Compute the slope and intercept: a, b
            a, b = np.polyfit(x, y, 1)

            # Print the result
            print('slope:', a, 'intercept:', b)




""" Chapter 2: Bootstrap confidence intervals

To "pull yourself up by your bootstraps" is a classic idiom meaning that you achieve a difficult task by yourself
with no help at all. In statistical inference, you want to know what would happen if you could repeat 
your data acquisition an infinite number of times. This task is impossible, but can we use only the data 
we actually have to get close to the same result as an infinitude of experiments? 
The answer is yes! The technique to do it is aptly called bootstrapping. 
This chapter will introduce you to this extraordinarily powerful tool

"""
class Bootstrap:
    def __init__(self):
        pass

    def ecdf(self,data):
        return prep.ecdf(data)

    def Visualizing_bootstrap_samples(self,rainfall):

        for _ in range(50):
            # Generate bootstrap sample: bs_sample
            bs_sample = np.random.choice(rainfall, size=len(rainfall))

            # Compute and plot ECDF from bootstrap sample
            x, y = self.ecdf(bs_sample)
            _ = plt.plot(x, y, marker='.', linestyle='none',
                        color='gray', alpha=0.1)

        # Compute and plot ECDF from original data
        x, y = self.ecdf(rainfall)
        _ = plt.plot(x, y, marker='.')

        # Make margins and label axes
        plt.margins(0.02)
        _ = plt.xlabel('yearly rainfall (mm)')
        _ = plt.ylabel('ECDF')

        # Show the plot
        plt.show()

    def bootstrap_replicate_1d(self, data, func):
        """Generate bootstrap replicate of 1D data."""    
        bs_sample = np.random.choice(data, len(data))
        return func(bs_sample)

    def draw_bs_reps(self, data, func, size=1):
        """ Generating many bootstrap replicates """

        # Initialize array of replicates: bs_replicates
        bs_replicates = np.empty(size = size)
        # Generate replicates
        for i in range(size):
            bs_replicates[i] = self.bootstrap_replicate_1d(data, func)

        return bs_replicates

    def Bootstrap_rep_of_mean_SEM(self, rainfall): 

        # Take 10,000 bootstrap replicates of the mean: bs_replicates
        bs_replicates = self.draw_bs_reps(rainfall, np.mean, size = 10000)

        # Compute and print SEM
        sem = np.std(rainfall) / np.sqrt(len(rainfall))
        print(sem)

        # Compute and print standard deviation of bootstrap replicates
        bs_std = np.std(bs_replicates)
        print(bs_std)

        # Make a histogram of the results
        _ = plt.hist(bs_replicates, bins=50, normed=True)
        _ = plt.xlabel('mean annual rainfall (mm)')
        _ = plt.ylabel('PDF')

        # Show the plot
        plt.show()

    def Bootstrap_replicates_of_other_statistics(self, rainfall): 
        # Generate 10,000 bootstrap replicates of the variance: bs_replicates
        bs_replicates = self.draw_bs_reps(rainfall, np.var, size = 10000)

        # Put the variance in units of square centimeters
        bs_replicates = bs_replicates / 100

        # Make a histogram of the results
        _ = plt.hist(bs_replicates, bins=50, normed=True)
        _ = plt.xlabel('variance of annual rainfall (sq. cm)')
        _ = plt.ylabel('PDF')

        # Show the plot
        plt.show()

    def Bootstrap_nohitter(self, nohitter_times ):
        """ Confidence interval on the rate of no-hitters """
        # Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
        bs_replicates = self.draw_bs_reps(nohitter_times,np.mean, size = 10000)

        # Compute the 95% confidence interval: conf_int
        conf_int = np.percentile(bs_replicates, [2.5, 97.5])

        # Print the confidence interval
        print('95% confidence interval =', conf_int[1], 'games')

        # Plot the histogram of the replicates
        _ = plt.hist(bs_replicates, bins=50, normed=True)
        _ = plt.xlabel(r'$\tau$ (games)')
        _ = plt.ylabel('PDF')

        # Show the plot
        plt.show()

    def draw_bs_pairs_linreg(self, x, y, size=1):
        """Perform pairs bootstrap for linear regression."""

        # Set up array of indices to sample from: inds
        inds = np.arange(len(x))

        # Initialize replicates: bs_slope_reps, bs_intercept_reps
        bs_slope_reps = np.empty(size)
        bs_intercept_reps = np.empty(size)

        # Generate replicates
        for i in range(size):
            bs_inds = np.random.choice(inds, size=len(inds))
            bs_x, bs_y = x[bs_inds], y[bs_inds]
            bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

        return bs_slope_reps, bs_intercept_reps

    def Pair_bootstrap_data(self, fertility, illiteracy):
        # Generate replicates of slope and intercept using pairs bootstrap
        bs_slope_reps, bs_intercept_reps = self.draw_bs_pairs_linreg(illiteracy, fertility, 1000)

        # Compute and print 95% CI for slope
        print(np.percentile(bs_slope_reps, [2.5, 97.5]))

        # Plot the histogram
        _ = plt.hist(bs_slope_reps, bins=50, normed=True)
        _ = plt.xlabel('slope')
        _ = plt.ylabel('PDF')
        plt.show()

        """ Plotting bootstrap regressions """
        # Generate array of x-values for bootstrap lines: x
        x = np.array([0, 100])

        # Plot the bootstrap lines
        for i in range(100):
            _ = plt.plot(x, 
                        bs_slope_reps[i]*x + bs_intercept_reps[i],
                        linewidth=0.5, alpha=0.2, color='red')

        # Plot the data
        _ = plt.plot(illiteracy, fertility ,marker='.',linestyle='none')

        # Label axes, set the margins, and show the plot
        _ = plt.xlabel('illiteracy')
        _ = plt.ylabel('fertility')
        plt.margins(0.02)
        plt.show()



""" Chapter 3: Introduction to hypothesis testing

You now know how to define and estimate parameters given a model. But the question remains: 
how reasonable is it to observe your data if a model is true? This question is addressed by hypothesis tests. 
They are the icing on the inference cake. After completing this chapter, 
you will be able to carefully construct and test hypotheses using hacker statistics.

"""
class Hypothesis():
    def __init__(self):
        pass


    def permutation_sample(self, data1, data2):
        """Generate a permutation sample from two data sets."""

        # Concatenate the data sets: data
        data = np.concatenate((data1, data2))

        # Permute the concatenated array: permuted_data
        permuted_data = np.random.permutation(data)

        # Split the permuted array into two: perm_sample_1, perm_sample_2
        perm_sample_1 = permuted_data[:len(data1)]
        perm_sample_2 = permuted_data[len(data1):]

        return perm_sample_1, perm_sample_2

    
    def Visualizing_permutation_sampling(self, rain_june, rain_november, ecdf):

        for _ in range(50):
            # Generate permutation samples
            perm_sample_1, perm_sample_2 = self.permutation_sample(rain_june, rain_november)

            # Compute ECDFs
            x_1, y_1 = ecdf(perm_sample_1)
            x_2, y_2 = ecdf(perm_sample_2)

            # Plot ECDFs of permutation sample
            _ = plt.plot(x_1, y_1 , marker='.', linestyle='none',
                        color='red', alpha=0.02)
            _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                        color='blue', alpha=0.02)

        # Create and plot ECDFs from original data
        x_1, y_1 = ecdf(rain_june)
        x_2, y_2 = ecdf(rain_november)
        _ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
        _ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

        # Label axes, set margin, and show plot
        plt.margins(0.02)
        _ = plt.xlabel('monthly rainfall (mm)')
        _ = plt.ylabel('ECDF')
        plt.show()





""" Chapter 4: Hypothesis test examples

As you saw from the last chapter, hypothesis testing can be a bit tricky. You need to define the null hypothesis, 
figure out how to simulate it, and define clearly what it means to be "more extreme" to compute the p-value. 
Like any skill, practice makes perfect, and this chapter gives you some good practice with hypothesis tests.
"""


""" Chapter 5: Putting it all together: a case study

Every year for the past 40-plus years, Peter and Rosemary Grant have gone to the Galápagos island of Daphne Major 
and collected data on Darwin's finches. Using your skills in statistical inference, you will spend this chapter 
with their data, and witness first hand, through data, evolution in action. 
It's an exhilarating way to end the course!

"""

def main():

    # chap1 = Parameter()  
    # samples_std1   = np.random.normal(20, 1, size = 100000) 
    # chap1.test_ecdf_plot(samples_std1)
    # chap1.linear_regression(chap1.x, chap1.y)
    # chap1.linear_reg_on_Anscombe(chap1.anscombe_x, chap1.anscombe_y)
    
    



if __name__ == "__main__":
    main()