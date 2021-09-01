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

    
    def A_B_diff_test_3(self, illiteracy, fertility): 
        """ Hypothesis test on Pearson correlation """
        # Compute observed correlation: r_obs
        r_obs = self.pearson_r(illiteracy, fertility)

        # Initialize permutation replicates: perm_replicates
        perm_replicates = np.empty(10000)

        # Draw replicates
        for i in range(10000):
            # Permute illiteracy measurments: illiteracy_permuted
            illiteracy_permuted = np.random.permutation(illiteracy)

            # Compute Pearson correlation
            perm_replicates[i] = self.pearson_r(illiteracy_permuted, fertility)

        # Compute p-value: p
        p = np.sum(perm_replicates >= r_obs) / len(perm_replicates)
        print('p-val =', p)
    
    def A_B_diff_test_4(self, control, treated):
        # Compute x,y values for ECDFs
        x_control, y_control = self.ecdf(control)
        x_treated, y_treated = self.ecdf(treated)

        # Plot the ECDFs
        plt.plot(x_control, y_control, marker='.', linestyle='none')
        plt.plot(x_treated, y_treated, marker='.', linestyle='none')

        # Set the margins
        plt.margins(0.02)

        # Add a legend
        plt.legend(('control', 'treated'), loc='lower right')

        # Label axes and show plot
        plt.xlabel('millions of alive sperm per mL')
        plt.ylabel('ECDF')
        plt.show()

    def A_B_diff_test_5(self, control, treated, draw_bs_reps):
        """ Bootstrap hypothesis test on bee sperm counts """
        # Compute the difference in mean sperm count: diff_means
        diff_means = np.mean(control) - np.mean(treated)

        # Compute mean of pooled data: mean_count
        mean_count = np.mean(np.concatenate((control, treated)))

        # Generate shifted data sets
        control_shifted = control - np.mean(control) + mean_count
        treated_shifted = treated - np.mean(treated) + mean_count

        # Generate bootstrap replicates
        bs_reps_control = draw_bs_reps(control_shifted,
                            np.mean, size=10000)
        bs_reps_treated = draw_bs_reps(treated_shifted,
                            np.mean, size=10000)

        # Get replicates of difference of means: bs_replicates
        bs_replicates = bs_reps_control - bs_reps_treated

        # Compute and print p-value: p
        p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
                    / len(bs_replicates)
        print('p-value =', p)
 

""" Chapter 5: Putting it all together: a case study

Every year for the past 40-plus years, Peter and Rosemary Grant have gone to the Galápagos island of Daphne Major 
and collected data on Darwin's finches. Using your skills in statistical inference, you will spend this chapter 
with their data, and witness first hand, through data, evolution in action. 
It's an exhilarating way to end the course!

"""


from IPython.display import display
class Darwin_finches(Hypothesis):
    def __init__(self):
        super().__init__()
        darwin_data_a = pd.read_csv('finch_beaks_1975.csv',  header=0)
        darwin_data_b = pd.read_csv('finch_beaks_2012.csv',  header=0)

        print(darwin_data_a.info)
        print(darwin_data_b.info)

        darwin_data_a.rename({'Beak length, mm': 'blength', 'Beak depth, mm': 'bdepth'}, axis=1, inplace=True)


        # initialise data of lists.
        data_bdepth = {'beak_depth':darwin_data_a.bdepth.to_list() + darwin_data_b.bdepth.to_list(),
                'year': [1975]*len(darwin_data_a.bdepth.to_list()) + [2012]*len(darwin_data_b.bdepth.to_list()) }

        data_blength = {'beak_length':darwin_data_a.blength.to_list() + darwin_data_b.blength.to_list(),
                'year': [1975]*len(darwin_data_a.blength.to_list()) + [2012]*len(darwin_data_b.blength.to_list()) }
 
        # Create DataFrame
        self.df_bdepth = pd.DataFrame(data_bdepth)
        self.df_blength = pd.DataFrame(data_blength)

        # print(df_bdepth.info(), '\n', df_bdepth.head())

    def create_bee_swarm_plot(self, df):
        # Create bee swarm plot
        _ = sns.swarmplot(x = 'year', y = 'beak_depth',data= df)

        # Label the axes
        _ = plt.xlabel('year')
        _ = plt.ylabel('beak depth (mm)')

        # Show the plot
        plt.show()

    def ecdf_of_beak_depths(self, ecdf, bd_1975, bd_2012):
        """ The differences are much clearer in the ECDF. 
        The mean is larger in the 2012 data, and the variance does appear larger as well """
        # Compute ECDFs
        x_1975, y_1975 = ecdf(bd_1975)
        x_2012, y_2012 = ecdf(bd_2012)

        # Plot the ECDFs
        _ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
        _ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

        # Set margins
        plt.margins(0.02)

        # Add axis labels and legend
        _ = plt.xlabel('beak depth (mm)')
        _ = plt.ylabel('ECDF')
        _ = plt.legend(('1975', '2012'), loc='lower right')

        plt.show()

    def parameter_estimation(self, bd_1975, bd_2012, draw_bs_reps):
        """ Estimate the difference of the mean beak depth of the G. 
        scandens samples from 1975 and 2012 and report a 95% confidence interval."""
        # Compute mean of all beak depths
        mean_bd = np.mean(bd_1975) + np.mean(bd_2012)

        # Compute the difference of the sample means: mean_diff
        mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

        # Get bootstrap replicates of means
        bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
        bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

        # Compute samples of difference of means: bs_diff_replicates
        bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

        # Compute 95% confidence interval: conf_int
        conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

        # Print the results
        print("""
        Mean of beak depths (1975): {0:.3f} mm
        Mean of beak depths (2012): {1:.3f} mm
        Mean difference: {2:.3f} mm
        95% confidence interval: [{3:.3f}, {4:.3f}] mm""".format(
            np.mean(bd_1975), np.mean(bd_2012), mean_diff, *conf_int))
    
    def hypothesis_test_finches_depth(self, bd_2012, bd_1975, draw_bs_reps):
        """ Hypothesis test: Are beaks deeper in 2012 ? 
        We get a p-value of 0.0034, which suggests that there is a statistically significant difference. 
        But remember: it is very important to know how different they are! In the previous exercise, you got a difference of 0.2 mm between the means. 
        You should combine this with the statistical significance. Changing by 0.2 mm in 37 years is substantial by evolutionary standards. 
        If it kept changing at that rate, the beak depth would double in only 400"""

        # Compute the difference of the sample means: mean_diff
        mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

        # Compute mean of combined data set: combined_mean
        combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

        # Shift the samples
        bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
        bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

        # Get bootstrap replicates of shifted data sets
        bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, 10000)
        bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, 10000)

        # Compute replicates of difference of means: bs_diff_replicates
        bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

        # Compute the p-value
        p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

        # Print p-value
        print('p =', p)

    def EDA_of_beak_length_and_depth(self, bl_1975, bd_1975, bl_2012, bd_2012):
        # Make scatter plot of 1975 data
        _ = plt.scatter(bl_1975, bd_1975, marker='.',
                    linestyle='None', color='blue', alpha=0.5)

        # Make scatter plot of 2012 data
        _ = plt.scatter(bl_2012, bd_2012, marker='.',
                    linestyle='None', color='red', alpha=0.5)

        # Label axes and make legend
        _ = plt.xlabel('beak length (mm)')
        _ = plt.ylabel('beak depth (mm)')
        _ = plt.legend(('1975', '2012'), loc='upper left')

        # Show the plot
        plt.show()

    def Linear_regressions(self, bl_1975, bd_1975, bl_2012, bd_2012, draw_bs_pairs_linreg):
        """ Perform a linear regression for both the 1975 and 2012 data """
        # Compute the linear regressions
        slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, 1)
        slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, 1)

        # Perform pairs bootstrap for the linear regressions
        bs_slope_reps_1975, bs_intercept_reps_1975 = \
                draw_bs_pairs_linreg(bl_1975, bd_1975, 1000)
        bs_slope_reps_2012, bs_intercept_reps_2012 = \
                draw_bs_pairs_linreg(bl_2012, bd_2012, 1000)

        # Compute confidence intervals of slopes
        slope_conf_int_1975 =  np.percentile(bs_slope_reps_1975, [2.5, 97.5])
        slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
        intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])

        intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])


        # Print the results
        print('1975: slope =', slope_1975,
            'conf int =', slope_conf_int_1975)
        print('1975: intercept =', intercept_1975,
            'conf int =', intercept_conf_int_1975)
        print('2012: slope =', slope_2012,
            'conf int =', slope_conf_int_2012)
        print('2012: intercept =', intercept_2012,
            'conf int =', intercept_conf_int_2012)

    
    def Displaying_linear_regression_results (self, bl_1975, bd_1975, bl_2012, bd_2012, bs_slope_reps_1975, bs_intercept_reps_1975, bs_slope_reps_2012, bs_intercept_reps_2012):
        # Make scatter plot of 1975 data
        _ = plt.plot(bl_1975, bd_1975, marker='.',
                    linestyle='none', color='blue', alpha=0.5)

        # Make scatter plot of 2012 data
        _ = plt.plot(bl_2012, bd_2012, marker='.',
                    linestyle='none', color='red', alpha=0.5)

        # Label axes and make legend
        _ = plt.xlabel('beak length (mm)')
        _ = plt.ylabel('beak depth (mm)')
        _ = plt.legend(('1975', '2012'), loc='upper left')

        # Generate x-values for bootstrap lines: x
        x = np.array([10, 17])

        # Plot the bootstrap lines
        for i in range(100):
            plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i],
                    linewidth=0.5, alpha=0.2, color='blue')
            plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i],
                    linewidth=0.5, alpha=0.2, color='red')

        # Draw the plot again
        plt.show()

    def Beak_length_to_depth_ratio(self, bl_1975, bd_1975, bl_2012, bd_2012, draw_bs_reps):
        # Compute length-to-depth ratios
        ratio_1975 = np.array(bl_1975/ bd_1975)
        ratio_2012 = np.array(bl_2012/ bd_2012)

        # Compute means
        mean_ratio_1975 = np.mean(ratio_1975)
        mean_ratio_2012 = np.mean(ratio_2012)

        # Generate bootstrap replicates of the means
        bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, 10000)
        bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, 10000)

        # Compute the 99% confidence intervals
        conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
        conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

        # Print the results
        print('1975: mean ratio =', mean_ratio_1975,
            'conf int =', conf_int_1975)
        print('2012: mean ratio =', mean_ratio_2012,
            'conf int =', conf_int_2012)

    





def main():

    # chap1 = Parameter()  
    # samples_std1   = np.random.normal(20, 1, size = 100000) 
    # chap1.test_ecdf_plot(samples_std1)
    # chap1.linear_regression(chap1.x, chap1.y)
    # chap1.linear_reg_on_Anscombe(chap1.anscombe_x, chap1.anscombe_y)

    # chap3 = Hypothesis()
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

    # chap3.A_B_diff_test_2(nht_dead, nht_live)

    illiteracy = np.array([ 9.5, 49.2,  1. , 11.2,  9.8, 60. , 50.2, 51.2,  0.6,  1. ,  8.5,
                            6.1,  9.8,  1. , 42.2, 77.2, 18.7, 22.8,  8.5, 43.9,  1. ,  1. ,
                            1.5, 10.8, 11.9,  3.4,  0.4,  3.1,  6.6, 33.7, 40.4,  2.3, 17.2,
                            0.7, 36.1,  1. , 33.2, 55.9, 30.8, 87.4, 15.4, 54.6,  5.1,  1.1,
                            10.2, 19.8,  0. , 40.7, 57.2, 59.9,  3.1, 55.7, 22.8, 10.9, 34.7,
                            32.2, 43. ,  1.3,  1. ,  0.5, 78.4, 34.2, 84.9, 29.1, 31.3, 18.3,
                            81.8, 39. , 11.2, 67. ,  4.1,  0.2, 78.1,  1. ,  7.1,  1. , 29. ,
                            1.1, 11.7, 73.6, 33.9, 14. ,  0.3,  1. ,  0.8, 71.9, 40.1,  1. ,
                            2.1,  3.8, 16.5,  4.1,  0.5, 44.4, 46.3, 18.7,  6.5, 36.8, 18.6,
                            11.1, 22.1, 71.1,  1. ,  0. ,  0.9,  0.7, 45.5,  8.4,  0. ,  3.8,
                            8.5,  2. ,  1. , 58.9,  0.3,  1. , 14. , 47. ,  4.1,  2.2,  7.2,
                            0.3,  1.5, 50.5,  1.3,  0.6, 19.1,  6.9,  9.2,  2.2,  0.2, 12.3,
                            4.9,  4.6,  0.3, 16.5, 65.7, 63.5, 16.8,  0.2,  1.8,  9.6, 15.2,
                            14.4,  3.3, 10.6, 61.3, 10.9, 32.2,  9.3, 11.6, 20.7,  6.5,  6.7,
                            3.5,  1. ,  1.6, 20.5,  1.5, 16.7,  2. ,  0.9])
    fertility = np.array([1.769, 2.682, 2.077, 2.132, 1.827, 3.872, 2.288, 5.173, 1.393,
                        1.262, 2.156, 3.026, 2.033, 1.324, 2.816, 5.211, 2.1  , 1.781,
                        1.822, 5.908, 1.881, 1.852, 1.39 , 2.281, 2.505, 1.224, 1.361,
                        1.468, 2.404, 5.52 , 4.058, 2.223, 4.859, 1.267, 2.342, 1.579,
                        6.254, 2.334, 3.961, 6.505, 2.53 , 2.823, 2.498, 2.248, 2.508,
                        3.04 , 1.854, 4.22 , 5.1  , 4.967, 1.325, 4.514, 3.173, 2.308,
                        4.62 , 4.541, 5.637, 1.926, 1.747, 2.294, 5.841, 5.455, 7.069,
                        2.859, 4.018, 2.513, 5.405, 5.737, 3.363, 4.89 , 1.385, 1.505,
                        6.081, 1.784, 1.378, 1.45 , 1.841, 1.37 , 2.612, 5.329, 5.33 ,
                        3.371, 1.281, 1.871, 2.153, 5.378, 4.45 , 1.46 , 1.436, 1.612,
                        3.19 , 2.752, 3.35 , 4.01 , 4.166, 2.642, 2.977, 3.415, 2.295,
                        3.019, 2.683, 5.165, 1.849, 1.836, 2.518, 2.43 , 4.528, 1.263,
                        1.885, 1.943, 1.899, 1.442, 1.953, 4.697, 1.582, 2.025, 1.841,
                        5.011, 1.212, 1.502, 2.516, 1.367, 2.089, 4.388, 1.854, 1.748,
                        2.978, 2.152, 2.362, 1.988, 1.426, 3.29 , 3.264, 1.436, 1.393,
                        2.822, 4.969, 5.659, 3.24 , 1.693, 1.647, 2.36 , 1.792, 3.45 ,
                        1.516, 2.233, 2.563, 5.283, 3.885, 0.966, 2.373, 2.663, 1.251,
                        2.052, 3.371, 2.093, 2.   , 3.883, 3.852, 3.718, 1.732, 3.928])

    # chap3.A_B_diff_test_3(illiteracy, fertility)

    bd_1975 = np.array([ 8.4 ,  8.8 ,  8.4 ,  8.  ,  7.9 ,  8.9 ,  8.6 ,  8.5 ,  8.9 ,
                        9.1 ,  8.6 ,  9.8 ,  8.2 ,  9.  ,  9.7 ,  8.6 ,  8.2 ,  9.  ,
                        8.4 ,  8.6 ,  8.9 ,  9.1 ,  8.3 ,  8.7 ,  9.6 ,  8.5 ,  9.1 ,
                        9.  ,  9.2 ,  9.9 ,  8.6 ,  9.2 ,  8.4 ,  8.9 ,  8.5 , 10.4 ,
                        9.6 ,  9.1 ,  9.3 ,  9.3 ,  8.8 ,  8.3 ,  8.8 ,  9.1 , 10.1 ,
                        8.9 ,  9.2 ,  8.5 , 10.2 , 10.1 ,  9.2 ,  9.7 ,  9.1 ,  8.5 ,
                        8.2 ,  9.  ,  9.3 ,  8.  ,  9.1 ,  8.1 ,  8.3 ,  8.7 ,  8.8 ,
                        8.6 ,  8.7 ,  8.  ,  8.8 ,  9.  ,  9.1 ,  9.74,  9.1 ,  9.8 ,
                        10.4 ,  8.3 ,  9.44,  9.04,  9.  ,  9.05,  9.65,  9.45,  8.65,
                        9.45,  9.45,  9.05,  8.75,  9.45,  8.35])
    bd_2012 = np.array([ 9.4 ,  8.9 ,  9.5 , 11.  ,  8.7 ,  8.4 ,  9.1 ,  8.7 , 10.2 ,
                        9.6 ,  8.85,  8.8 ,  9.5 ,  9.2 ,  9.  ,  9.8 ,  9.3 ,  9.  ,
                        10.2 ,  7.7 ,  9.  ,  9.5 ,  9.4 ,  8.  ,  8.9 ,  9.4 ,  9.5 ,
                        8.  , 10.  ,  8.95,  8.2 ,  8.8 ,  9.2 ,  9.4 ,  9.5 ,  8.1 ,
                        9.5 ,  8.4 ,  9.3 ,  9.3 ,  9.6 ,  9.2 , 10.  ,  8.9 , 10.5 ,
                        8.9 ,  8.6 ,  8.8 ,  9.15,  9.5 ,  9.1 , 10.2 ,  8.4 , 10.  ,
                        10.2 ,  9.3 , 10.8 ,  8.3 ,  7.8 ,  9.8 ,  7.9 ,  8.9 ,  7.7 ,
                        8.9 ,  9.4 ,  9.4 ,  8.5 ,  8.5 ,  9.6 , 10.2 ,  8.8 ,  9.5 ,
                        9.3 ,  9.  ,  9.2 ,  8.7 ,  9.  ,  9.1 ,  8.7 ,  9.4 ,  9.8 ,
                        8.6 , 10.6 ,  9.  ,  9.5 ,  8.1 ,  9.3 ,  9.6 ,  8.5 ,  8.2 ,
                        8.  ,  9.5 ,  9.7 ,  9.9 ,  9.1 ,  9.5 ,  9.8 ,  8.4 ,  8.3 ,
                        9.6 ,  9.4 , 10.  ,  8.9 ,  9.1 ,  9.8 ,  9.3 ,  9.9 ,  8.9 ,
                        8.5 , 10.6 ,  9.3 ,  8.9 ,  8.9 ,  9.7 ,  9.8 , 10.5 ,  8.4 ,
                        10.  ,  9.  ,  8.7 ,  8.8 ,  8.4 ,  9.3 ,  9.8 ,  8.9 ,  9.8 ,
                        9.1 ])
    bl_1975 = np.array([13.9 , 14.  , 12.9 , 13.5 , 12.9 , 14.6 , 13.  , 14.2 , 14.  ,
            14.2 , 13.1 , 15.1 , 13.5 , 14.4 , 14.9 , 12.9 , 13.  , 14.9 ,
            14.  , 13.8 , 13.  , 14.75, 13.7 , 13.8 , 14.  , 14.6 , 15.2 ,
            13.5 , 15.1 , 15.  , 12.8 , 14.9 , 15.3 , 13.4 , 14.2 , 15.1 ,
            15.1 , 14.  , 13.6 , 14.  , 14.  , 13.9 , 14.  , 14.9 , 15.6 ,
            13.8 , 14.4 , 12.8 , 14.2 , 13.4 , 14.  , 14.8 , 14.2 , 13.5 ,
            13.4 , 14.6 , 13.5 , 13.7 , 13.9 , 13.1 , 13.4 , 13.8 , 13.6 ,
            14.  , 13.5 , 12.8 , 14.  , 13.4 , 14.9 , 15.54, 14.63, 14.73,
            15.73, 14.83, 15.94, 15.14, 14.23, 14.15, 14.35, 14.95, 13.95,
            14.05, 14.55, 14.05, 14.45, 15.05, 13.25])
    bl_2012 = np.array([14.3 , 12.5 , 13.7 , 13.8 , 12.  , 13.  , 13.  , 13.6 , 12.8 ,
                        13.6 , 12.95, 13.1 , 13.4 , 13.9 , 12.3 , 14.  , 12.5 , 12.3 ,
                        13.9 , 13.1 , 12.5 , 13.9 , 13.7 , 12.  , 14.4 , 13.5 , 13.8 ,
                        13.  , 14.9 , 12.5 , 12.3 , 12.8 , 13.4 , 13.8 , 13.5 , 13.5 ,
                        13.4 , 12.3 , 14.35, 13.2 , 13.8 , 14.6 , 14.3 , 13.8 , 13.6 ,
                        12.9 , 13.  , 13.5 , 13.2 , 13.7 , 13.1 , 13.2 , 12.6 , 13.  ,
                        13.9 , 13.2 , 15.  , 13.37, 11.4 , 13.8 , 13.  , 13.  , 13.1 ,
                        12.8 , 13.3 , 13.5 , 12.4 , 13.1 , 14.  , 13.5 , 11.8 , 13.7 ,
                        13.2 , 12.2 , 13.  , 13.1 , 14.7 , 13.7 , 13.5 , 13.3 , 14.1 ,
                        12.5 , 13.7 , 14.6 , 14.1 , 12.9 , 13.9 , 13.4 , 13.  , 12.7 ,
                        12.1 , 14.  , 14.9 , 13.9 , 12.9 , 14.6 , 14.  , 13.  , 12.7 ,
                        14.  , 14.1 , 14.1 , 13.  , 13.5 , 13.4 , 13.9 , 13.1 , 12.9 ,
                        14.  , 14.  , 14.1 , 14.7 , 13.4 , 13.8 , 13.4 , 13.8 , 12.4 ,
                        14.1 , 12.9 , 13.9 , 14.3 , 13.2 , 14.2 , 13.  , 14.6 , 13.1 , 15.2 ])


    chap5 = Darwin_finches()
    # chap5.create_bee_swarm_plot(chap5.df_bdepth)
    # chap5.ecdf_of_beak_depths( prep.ecdf ,bd_1975, bd_2012)
    chap5.EDA_of_beak_length_and_depth(bl_1975, bd_1975, bl_2012, bd_2012)
    


    


if __name__ == "__main__":
    main()