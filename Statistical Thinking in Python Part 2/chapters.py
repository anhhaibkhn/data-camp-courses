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

import sys
sys.path.insert(0, 'E:/data_science_resources/git_data_camp/data-camp-courses/Statistical Thinking in Python Part 1/')
from chapter_1_2_3_4 import Preparing_data as prep


class Parameter:
    def __init__(self):
        
        pass

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

        





def main():

    chap1 = Parameter()  
    # samples_std1   = np.random.normal(20, 1, size = 100000) 
    # chap1.test_ecdf_plot(samples_std1)
    

    
    



if __name__ == "__main__":
    main()