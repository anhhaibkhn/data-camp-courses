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
import seaborn as sns
import os


import sys
mydir =  sys.path[0]
# import module from an upper directory 
newdir = os.path.abspath(os.path.join(mydir, '..')) + '\Statistical Thinking in Python Part 1'
print(newdir)
sys.path.insert(0, newdir)
from chapter_1_2_3_4 import Preparing_data as prep


class Parameter:
    def __init__(self):
        
        self.anscombe_data = pd.read_csv("anscombe.csv")
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
        bs_replicates = np.empty(size)
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
class Hypothesis(Bootstrap):
    def __init__(self):
        super().__init__()
        current_file = os.path.abspath(os.path.dirname(__file__))
        #csv_filename
        csv_filename = os.path.join(current_file, 'frog_tongue.csv')
        self.frog_tongue = pd.read_csv(csv_filename,  skiprows= 14)
        
        self.frog_tongue.rename(columns={'ID': 'OLD_ID'}, inplace=True)
        # Create the dictionary
        ID_dict  ={'I' : 'C', 'II' : 'A', 'III' : 'D', 'IV': 'B'}   
        # Add a new column named 'ID'
        self.frog_tongue['ID'] = self.frog_tongue['OLD_ID'].map(ID_dict)
        self.frog_tongue['impact_force'] = self.frog_tongue['impact force (mN)'] / 1000

        # print(self.frog_tongue.head())
        # print(self.frog_tongue.info())

        options = ['A', 'B'] 
        # selecting rows based on condition
        self.frog_df = self.frog_tongue.loc[self.frog_tongue['ID'].isin(options), ['ID', 'impact_force']]
        
        # print('\nResult dataframe :\n', self.frog_df)
        self.force_a = self.frog_df.loc[self.frog_df['ID'] == 'A', 'impact_force'].to_numpy().astype(np.float64)
        self.force_b = self.frog_df.loc[self.frog_df['ID'] == 'B', 'impact_force'].to_numpy().astype(np.float64)
 

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

    def draw_perm_reps(self, data_1, data_2, func, size=1):
        """Generate multiple permutation replicates."""

        # Initialize array of replicates: perm_replicates
        perm_replicates = np.empty(size)

        for i in range(size):
            # Generate permutation sample
            perm_sample_1, perm_sample_2 = self.permutation_sample(data_1, data_2)

            # Compute the test statistic
            perm_replicates[i] = func(perm_sample_1, perm_sample_2)

        return perm_replicates

    def make_bee_swarm_plot(self, df):
        """ Make bee swarm plot """
        _ = sns.swarmplot('ID', y = 'impact_force', data=df)

        # Label the axes
        _ = plt.xlabel('frog')
        _ = plt.ylabel('impact force (N)')

        # Show the plot
        plt.show()

    
    def permutation_test_on_frog_data(self,force_a, force_b):

        def diff_of_means(data_1, data_2):
            """Difference in means of two arrays."""
            # The difference of means of data_1, data_2: diff
            diff = np.mean(data_1) - np.mean(data_2)

            return diff

        # Compute difference of mean impact force from experiment: empirical_diff_means
        empirical_diff_means = diff_of_means(force_a, force_b)

        # Draw 10,000 permutation replicates: perm_replicates
        perm_replicates = self.draw_perm_reps(force_a, force_b,
                                        diff_of_means , size=10000)

        # Compute p-value: p
        p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

        # Print the result
        print('p-value =', p)

    def one_sample_bootstrap_hypothesis_test(self, force_b):

        # Make an array of translated impact forces: translated_force_b
        translated_force_b = force_b - np.mean(force_b) + 0.55

        # Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
        bs_replicates = self.draw_bs_reps(translated_force_b, np.mean, 10000)

        # Compute fraction of replicates that are less than the observed Frog B force: p
        p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

        # Print the p-value
        print('p = ', p)

    def two_sample_bootstrap_hypothesis_test(self, force_a, force_b, forces_concat, empirical_diff_means):
        # Compute mean of all forces: mean_force
        mean_force = np.mean(forces_concat)

        # Generate shifted arrays
        force_a_shifted = force_a - np.mean(force_a) + mean_force
        force_b_shifted = force_b - np.mean(force_b) + mean_force 

        # Compute 10,000 bootstrap replicates from shifted arrays
        bs_replicates_a = self.draw_bs_reps(force_a_shifted, np.mean, 10000)
        bs_replicates_b = self.draw_bs_reps(force_b_shifted, np.mean, 10000)

        # Get replicates of difference of means: bs_replicates
        bs_replicates = bs_replicates_a - bs_replicates_b

        # Compute and print p-value: p
        p = np.sum(bs_replicates >= empirical_diff_means ) / 10000
        print('p-value =', p)



    """ Chapter 4: Hypothesis test examples

    As you saw from the last chapter, hypothesis testing can be a bit tricky. You need to define the null hypothesis, 
    figure out how to simulate it, and define clearly what it means to be "more extreme" to compute the p-value. 
    Like any skill, practice makes perfect, and this chapter gives you some good practice with hypothesis tests.
    """
    def diff_frac(self, data_A, data_B):        
        frac_A = np.sum(data_A) / len(data_A)        
        frac_B = np.sum(data_B) / len(data_B)
        
        return frac_B - frac_A

    def calculate_p_value(self, data_A, data_B):
        diff_frac_obs = self.diff_frac(data_A, data_B)

        perm_replicates = self.draw_bs_reps(data_A, data_B, self.diff_frac)
        
        p_value = np.sum(perm_replicates >= diff_frac_obs) / 10000
            
        return p_value

    def A_B_diff_test_1(self):  
        # Construct arrays of data: dems, reps
        dems = np.array([True] * 153 + [False] * 91)
        reps = np.array([True] * 136 + [False] * 35)

        def frac_yea_dems(dems, reps):
            """Compute fraction of Democrat yea votes."""
            frac = np.sum(dems) / len(dems)
            return frac

        # Acquire permutation samples: perm_replicates
        perm_replicates = self.draw_perm_reps(dems, reps, frac_yea_dems, 10000)

        # Compute and print p-value: p
        p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
        print('p-value =', p)
  

    def diff_of_means(self, data_1, data_2):
            """Difference in means of two arrays."""
            # The difference of means of data_1, data_2: diff
            diff = np.mean(data_1) - np.mean(data_2)

            return diff

            
    def A_B_diff_test_2(self, nht_dead, nht_live):
        # Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
        nht_diff_obs = self.diff_of_means(nht_dead, nht_live)

        # Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
        perm_replicates = self.draw_perm_reps(nht_dead, nht_live, self.diff_of_means , 10000)


        # Compute and print the p-value: p
        p = np.sum(perm_replicates <= nht_diff_obs) / len(perm_replicates)
        print('p-val =', p)





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

    chap3 = Hypothesis()
    # chap3.make_bee_swarm_plot(chap3.frog_df)
    # chap3.permutation_test_on_frog_data(chap3.force_a, chap3.force_b)
    # chap3.one_sample_bootstrap_hypothesis_test(chap3.force_b)
 
    nht_dead = np.array([  -1,  894,   10,  130,    1,  934,   29,    6,  485,  254,  372,
                            81,  191,  355,  180,  286,   47,  269,  361,  173,  246,  492,
                            462, 1319,   58,  297,   31, 2970,  640,  237,  434,  570,   77,
                            271,  563, 3365,   89,    0,  379,  221,  479,  367,  628,  843,
                            1613, 1101,  215,  684,  814,  278,  324,  161,  219,  545,  715,
                            966,  624,   29,  450,  107,   20,   91, 1325,  124, 1468,  104,
                            1309,  429,   62, 1878, 1104,  123,  251,   93,  188,  983,  166,
                            96,  702,   23,  524,   26,  299,   59,   39,   12,    2,  308,
                            1114,  813,  887])

    nht_live = np.array([ 645, 2088,   42, 2090,   11,  886, 1665, 1084, 2900, 2432,  750,
                        4021, 1070, 1765, 1322,   26,  548, 1525,   77, 2181, 2752,  127,
                        2147,  211,   41, 1575,  151,  479,  697,  557, 2267,  542,  392,
                        73,  603,  233,  255,  528,  397, 1529, 1023, 1194,  462,  583,
                        37,  943,  996,  480, 1497,  717,  224,  219, 1531,  498,   44,
                        288,  267,  600,   52,  269, 1086,  386,  176, 2199,  216,   54,
                        675, 1243,  463,  650,  171,  327,  110,  774,  509,    8,  197,
                        136,   12, 1124,   64,  380,  811,  232,  192,  731,  715,  226,
                        605,  539, 1491,  323,  240,  179,  702,  156,   82, 1397,  354,
                        778,  603, 1001,  385,  986,  203,  149,  576,  445,  180, 1403,
                        252,  675, 1351, 2983, 1568,   45,  899, 3260, 1025,   31,  100,
                        2055, 4043,   79,  238, 3931, 2351,  595,  110,  215,    0,  563,
                        206,  660,  242,  577,  179,  157,  192,  192, 1848,  792, 1693,
                        55,  388,  225, 1134, 1172, 1555,   31, 1582, 1044,  378, 1687,
                        2915,  280,  765, 2819,  511, 1521,  745, 2491,  580, 2072, 6450,
                        578,  745, 1075, 1103, 1549, 1520,  138, 1202,  296,  277,  351,
                        391,  950,  459,   62, 1056, 1128,  139,  420,   87,   71,  814,
                        603, 1349,  162, 1027,  783,  326,  101,  876,  381,  905,  156,
                        419,  239,  119,  129,  467])

    chap3.A_B_diff_test_2(nht_dead, nht_live)
    


    


if __name__ == "__main__":
    main()