'''          Chapter 2: Distributions

In the first chapter, having cleaned and validated your data, 
you began exploring it by using histograms to visualize distributions. 
In this chapter, you'll learn how to represent distributions using Probability Mass Functions (PMFs)
 and Cumulative Distribution Functions (CDFs). You'll learn when to use each of them, and why, 
while working with a new dataset obtained from the General Social Survey 

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from empiricaldist import Pmf
from empiricaldist import Cdf

class Distribution():
    ''' class to check the data distribution'''

    def __init__ (self):
        # display result: 0:hide  1:display
        self.paras_verbose = 1
        # load the General Social Survey dataset
        self.gss = pd.read_hdf('gss.hdf5', 'gss')

    # make some functions here 
    def verbose(self, msg ='\n', detail_msg ='\n'):
        ''' Verbose function for print information to stdout'''
        if self.paras_verbose == 1:           
            print('[INFO]', msg)
            print(detail_msg)

    def histogram(self, pd_series, str_label, x_str, y_str, bin_num):
        plt.hist(pd_series.dropna(), label = str_label, bins = bin_num)
        plt.xlabel(x_str)
        plt.ylabel(y_str)
        plt.show()

    def pmf_vs_histogram(self, msg):
        # first message
        self.verbose(msg,self.gss.head())

        # education histogram
        educ = self.gss['educ']
        self.histogram(educ, 'educ', 'Years of education', 'Count', 20)

        # PMF (Probability Mass Function)
        pmf_educ = Pmf.from_seq(educ, normalize = False)
        self.verbose('PMF: normalize = False', pmf_educ.head(5))
        print(pmf_educ[12])

        pmf_educ = Pmf.from_seq(educ, normalize = True)
        self.verbose('PMF: normalize = True', pmf_educ.head(5))
        print(pmf_educ[12])

        # pmf histogram function
        pmf_educ.bar(label = 'educ')
        plt.xlabel('Years of education')
        plt.ylabel('PMF')
        plt.show()

    def cdf_vs_pmf(self, msg):
        # first message
        self.verbose(msg,self.gss.head())
        age = self.gss['age']

        # check age PMF
        # pmf_age = Pmf.from_seq(age, normalize = True)
        # # plot age pmf
        # pmf_age.bar(label = 'Age')
        # plt.xlabel('Age')
        # plt.ylabel('PMF')
        # plt.show()

        # check age CDF
        cdf_age = Cdf.from_seq(age)
        # cdf_age.plot()
        # plt.xlabel('Age')
        # plt.ylabel('CDF')
        # plt.show()

        # evaluate the CDF and the inverse CDF
        q = 51 # age = 51
        print(cdf_age(q)) # age 51 has around 0.66

        p = 0.25 
        print(cdf_age.inverse(p)) # probability = 0.25 inverse to age = 30. 

        # check the IQR (interquartile range)



# main 
test = Distribution()
## PMF and Histogram 
# test.pmf_vs_histogram('begin chapter 2')

## CDF (Cumulative Distribution Functions)
test.cdf_vs_pmf('Cumulative distribution functions')