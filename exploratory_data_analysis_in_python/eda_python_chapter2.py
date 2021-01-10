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
from scipy.stats import norm 
import seaborn as sns 

class Distributions():
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
        pmf_age = Pmf.from_seq(age, normalize = True)
        # plot age pmf
        self.pmf_plot(pmf_age,'Age','Age','PMF')

        # check age CDF
        cdf_age = Cdf.from_seq(age)
        self.cdf_plot(cdf_age, 'Age', 'CDF')

        # evaluate the CDF and the inverse CDF
        q = 51 # age = 51
        self.verbose('cdf of q = 51: ',cdf_age(q)) # age 51 has around 0.66
        p = 0.25 
        self.verbose('cdf of p = 0.25: ',cdf_age.inverse(p)) # probability = 0.25 inverse to age = 30. 

        # check the IQR (interquartile range) difference of 0.75 and 0.25
        cdf_income = Cdf.from_seq(self.gss['realinc'])
        iqr_income = self.calculate_iqr(cdf_income)
        self.verbose('iqr_income is ', iqr_income )

        # plot income  cdf
        self.cdf_plot(cdf_income, 'Income (1986 USD)', 'CDF')

    
    def calculate_iqr(self, cdf_data):
        # Calculate the 75th percentile 
        percentile_75th = cdf_data.inverse(0.75)
        # Calculate the 25th percentile
        percentile_25th = cdf_data.inverse(0.25)
        # Calculate the interquartile range
        iqr = percentile_75th - percentile_25th
        # return the interquartile range
        return iqr

    def cdf_plot(self, cdf_data, x_label, y_label): 
        cdf_data.plot()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def pmf_plot(self, pmf_data,pmf_label, x_label, y_label): 
        # pmf plot as bar chart
        pmf_data.bar(label = pmf_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def pmf_comparing_distribution(self):
        # multiple PMFs
        male = self.gss['sex']  == 1
        age = self.gss['age']

        male_age = age[male]
        female_age = age[~male]

        Pmf.from_seq(male_age).plot(label = 'Male')
        Pmf.from_seq(female_age).plot(label = 'Female')
        plt.xlabel('Age (years)')
        plt.ylabel('PMF - Count')
        plt.show() # distribution looks quite noisy

    def cdf_comparing_distribution(self):
        # multiple CDFs
        male = self.gss['sex']  == 1
        age = self.gss['age']

        male_age = age[male]
        female_age = age[~male]

        Cdf.from_seq(male_age).plot(label = 'Male')
        Cdf.from_seq(female_age).plot(label = 'Female')
        plt.xlabel('Age (years)')
        plt.ylabel('CDF - Count')
        plt.show() # distribution looks very identical

    def pmf_income_distribution(self):
        income = self.gss['realinc']
        pre95 = self.gss['year'] < 1995

        Pmf.from_seq(income[pre95]).plot(label = 'Before 1995')
        Pmf.from_seq(income[~pre95]).plot(label = 'After 1995')
        plt.xlabel('Income (1986 USD)')
        plt.ylabel('PMF')
        plt.show() # too noisy, hard to tell

    def cdf_income_distribution(self):
        income = self.gss['realinc']
        pre95 = self.gss['year'] < 1995

        Cdf.from_seq(income[pre95]).plot(label = 'Before 1995')
        Cdf.from_seq(income[~pre95]).plot(label = 'After 1995')
        plt.xlabel('Income (1986 USD)')
        plt.ylabel('CDF')
        plt.show() # CDFs gave a Clearer picture than PMFs

    def cdf_education_distribution(self):
        educ = self.gss['educ']

        # Bachelor's degree
        bach = (educ >= 16)

        # Associate degree
        assc = (educ >= 14) & (educ < 16)

        # High school (12 or fewer years of education)
        high = (educ <= 12)
        print('High school has cdf = ',high.mean())

        # Plot income CDFs
        income = self.gss['realinc']

        # Plot the CDFs
        Cdf.from_seq(income[high]).plot(label='High school')
        Cdf.from_seq(income[assc]).plot(label='Associate')
        Cdf.from_seq(income[bach]).plot(label='Bachelor')

        # Label the axes
        plt.xlabel('Income (1986 USD)')
        plt.ylabel('CDF')
        plt.legend()
        plt.show() # higher education brings higher income 

    def modeling_distribution(self):
        # the normal distribution 
        sample = np.random.normal(size = 1000)
        # self.cdf_plot(Cdf.from_seq(sample),'Random value', 'CDF')

        # The normal CDF
        xs = np.linspace(-3, 3) # create an array
        
        # norm from scipy is object that represents the normal distribution 
        # ys = norm(0,1).cdf(xs) # ys: mean = zero, std = 1;  
        # plt.plot(xs, ys, color = 'gray')
        # self.cdf_plot(Cdf.from_seq(sample),'Random value', 'CDF')

        # The bell Curve
        # PDF : Probablility Densitiy Function 
        ys = norm(0,1).pdf(xs)
        plt.plot(xs, ys, color = 'gray')
        plt.xlabel('Normal Random value')
        plt.ylabel('PDF')
        # plt.show()

        # KDE plot
        self.KDE_plot(sample, 'Normal random value', 'Estimated probability density')

    # Kernel Density Estimation <a process to getting Pmf to Pdf>
    def KDE_plot(self, sample, x_label, y_label):
        sns.kdeplot(sample)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def income_distribution(self):
        # Extract realinc and compute its log
        income = self.gss['realinc']
        log_income = np.log10(income)

        # Compute mean and standard deviation
        mean = log_income.mean()
        std = log_income.std()
        self.verbose('mean of log10 gss income',mean)
        self.verbose('std of log10 gss income',std)

        # Make a norm object
        dist = norm(mean, std)

        # Comparing CDFs
        xs = np.linspace(2, 5.5)

        # Evaluate the model CDF
        ys = dist.cdf(xs)
        # Plot the model CDF
        plt.clf()
        plt.plot(xs, ys, color='gray')
        # Create and plot the Cdf of log_income
        self.cdf_plot(Cdf.from_seq(log_income),'log10 of realinc' ,'CDF' )

        # Evaluate the normal PDF
        ys = dist.pdf(xs)

        # Plot the model PDF
        plt.clf()
        plt.plot(xs, ys, color='gray')
        # Plot the data KDE
        self.KDE_plot(log_income,'log10 of realinc', 'PDF')
        



'''
Uncomment the below code parts (one by one) to explore the chapter 2 content
'''
# main 
# chap2 = Distributions()

## PMF and Histogram 
# test.pmf_vs_histogram('begin chapter 2')

## CDF (Cumulative Distribution Functions)
# test.cdf_vs_pmf('Cumulative distribution functions')

## PMF Comparing distribution 
# test.pmf_comparing_distribution()

## CDF Comparing distribution 
# test.cdf_comparing_distribution()

## PMF Income distribution
# test.pmf_income_distribution()

## CDF Income distribution 
# test.cdf_income_distribution()

## CDF education distribution
# test.cdf_education_distribution()

## Modeling distribution of income, comparing CDFs. 
# test.modeling_distribution()

## norm income distribution
# chap2.income_distribution()