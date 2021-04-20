""" Chapter 3 Additional Plot Types
Overview of more complex plot types included in Seaborn.

"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Seaborn_Intermediate():

    def __init__(self):
        self.bicycle_df = pd.read_csv("WashingtonDCBikeShare_bike_share.csv")
        self.bicycle_df = self.summary_df(self.bicycle_df )

        self.college_df  = pd.read_csv("college_datav3.csv")
        self.college_df = self.summary_df(self.college_df )

        self.daily_show_df  = pd.read_csv("daily_show_guests_cleaned.csv")
        self.daily_show_df = self.summary_df(self.daily_show_df )

        self.car_insurance_df  = pd.read_csv("insurance_premiums.csv")
        self.car_insurance_df = self.summary_df(self.car_insurance_df )

    def summary_df(self,df, show = False):
        df.dropna(how='all')
        df.drop_duplicates()
        if show:
            print("Summary of the basic information about this DataFrame and its data:")
            print(df.info())
            print("Top 5 rows:")
            print(df.head())
            print("Statistical data:")
            print(df.describe())
        
        return df


    def exercise_1(self):
        """Regression plot"""
        sns.regplot(data=self.bicycle_df, x='temp', y ='total_rentals', marker='+')
        plt.title('regression plot')
        plt.show()

        sns.residplot(data=self.bicycle_df, x='temp', y ='total_rentals')
        plt.title('residual plot')
        plt.show()

        sns.regplot(data=self.bicycle_df, x='temp', y ='total_rentals', order=2)
        plt.title('regression plot with polynomial = 2')
        plt.show()

        sns.regplot(data=self.bicycle_df, x='mnth', y ='total_rentals',
                                                    x_jitter=.1, order=2)
        plt.title('regression plot with x_jitter folliwng months')
        plt.show()

        sns.regplot(data=self.bicycle_df, x='mnth', y ='total_rentals',
                                                    x_estimator = np.mean, order=2)
        plt.title('regression plot with x_estimator = np.mean, folliwng months')
        plt.show()

        sns.regplot(data=self.bicycle_df, x='temp', y ='total_rentals',
                                                    x_bins= 12)
        plt.title('regression plot with Binning the data')
        plt.show()

    def exercise_2(self):
        # Display a regression plot for Tuition
        sns.regplot(data=self.college_df, y='Tuition',
                    x="SAT_AVG_ALL",
                    marker='^',
                    color='g')

        plt.title('regression plot for Tuition')
        plt.show()
        plt.clf()

        # Display the residual plot
        sns.residplot(data=self.college_df,
                y='Tuition',
                x="SAT_AVG_ALL",
                color='g')

        plt.title('residual plot for Tuition')
        plt.show()
        plt.clf()

    def exercise_3(self):
        # Plot a regression plot of Tuition and the Percentage of Pell Grants
        sns.regplot(data=self.college_df,
                    y='Tuition',
                    x="PCTPELL", x_bins=5, order = 2)
        plt.title('Plot a regression plot of Tuition and the Percentage of Pell Grants')
        plt.show()
        plt.clf()

    def reformat_df(self, df, index, col, values):
        """puting data to matrix form for heatmap func"""
        crt_df = pd.crosstab( index =df[index], columns=df[col], values = df[values], aggfunc='mean').round(0)
        print(crt_df)
        return crt_df


    def ex_4(self, df):
        """ matrix plot for heatmap """
        # reformat data
        df_crosstab= self.reformat_df(df, "mnth", "weekday", "total_rentals")
        
        # customize a heatmap 
        sns.heatmap(df_crosstab, annot= True,  fmt=".0f", cmap="YlGnBu", cbar=False, linewidths=.5)

        plt.title('build a heatmap for bike rentals')
        plt.show()
        plt.clf()
    
        # customize a heatmap 2
        sns.heatmap(df_crosstab, annot= True,  fmt=".0f", cmap="YlGnBu", cbar=False, linewidths=.5,
                            center=df_crosstab.loc[9,6])

        plt.title('customize the center area')
        plt.show()
        plt.clf()
        
        sns.heatmap(df_crosstab.corr())
        plt.title('Plottingacorrelationmatrix')
        plt.show()
        plt.clf()

    def ex_5(self, df):
        """ creating heatmap """
        # Create a crosstab table of the data
        pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])
        print(pd_crosstab)

        # Plot a heatmap of the table
        sns.heatmap(pd_crosstab)

        # Rotate tick marks for visibility
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        plt.title('heatmap with no customize')
        plt.show()

        # Plot a heatmap of the table with no color bar and using the BuGn palette
        sns.heatmap(pd_crosstab, cbar=False, cmap="BuGn", linewidths=0.3)
        plt.title('heatmap with customization')
        plt.show()

    """ chapter 4 Creating Plots on Data Aware Grids
    Using Seaborn to draw multiple plots in a single figure."""

    def ex_6(self, df):
        """ create FacetGrid 
        This plots tell us a lot about the relationships 
        between Average SAT scores by Degree Types offered at a university.
        """
        # Create FacetGrid with Degree_Type and specify the order of the rows using row_order
        g2 = sns.FacetGrid(df, 
                    row="Degree_Type",
                    row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

        # Map a pointplot of SAT_AVG_ALL onto the grid
        g2.map(sns.pointplot, 'SAT_AVG_ALL')

        # Show the plot
        plt.show()
        plt.clf()

    def ex_7(self, df):
        """ The factorplot is often more convenient than using a FacetGrid 
        for creating data aware grids.
        """
        # Create a facetted pointplot of Average SAT_AVG_ALL scores facetted by Degree Type 
        sns.factorplot(data=df,
                x='SAT_AVG_ALL',
                kind='point',
                row='Degree_Type',
                row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])
        plt.title('factorplot with customization')
        plt.show()
        plt.clf()

    def ex_8(self, df):
        """ Using a lmplot
        """
        # Create a FacetGrid varying by column and columns ordered with the degree_order variable
        degree_ord = ['Graduate', 'Bachelors', 'Associates']
        g = sns.FacetGrid(df, col="Degree_Type", col_order= degree_ord)

        # Map a scatter plot of Undergrad Population compared to PCTPELL
        g.map(plt.scatter, 'UG', 'PCTPELL')
        plt.title('FacetGrid varying by column and columns ordered')
        plt.show()
        plt.clf()

        # Re-create the plot above as an lmplot
        sns.lmplot(data=df,
                x='UG',
                y='PCTPELL',
                col="Degree_Type",
                col_order=degree_ord)
        plt.show()
        plt.clf()

        # Create an lmplot that has a column for Ownership, a row for Degree_Type and hue based on the WOMENONLY column
        inst_ord = ['Public', 'Private non-profit']

        sns.lmplot(data=df,
                x='SAT_AVG_ALL',
                y='Tuition',
                col="Ownership",
                row='Degree_Type',
                row_order=['Graduate', 'Bachelors'],
                hue='WOMENONLY',
                col_order=inst_ord)
        plt.show()
        plt.clf()
    
    def ex_9(self, df):
        """ Pair Grid and pair plot """
        # Create a PairGrid with a scatter plot for fatal_collisions and premiums
        g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
        g2 = g.map(plt.scatter)

        plt.show()
        plt.clf()


    def ex_10(self, df):
        """ Join Grid and join plot """
        # Create the same PairGrid but map a histogram on the diag
        g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
        g2 = g.map_diag(plt.hist)
        g3 = g2.map_offdiag(plt.scatter)

        plt.show()
        plt.clf()

    def ex_11(self, df):
        """ Using a pairplot """
        # Create a pairwise plot of the variables using a scatter plot
        sns.pairplot(data=df,
                vars=["fatal_collisions", "premiums"],
                kind='scatter')

        plt.show()
        plt.clf()

    def ex_12(self, df):
        # Create another pairplot using the "Region" to color code the results.
        # Plot the same data but use a different color palette and color code by Region
        sns.pairplot(data=df,
                vars=["fatal_collisions", "premiums"],
                kind='scatter',
                hue='Region',
                palette='RdBu',
                diag_kws={'alpha':.5})

        plt.show()
        plt.clf()

def main():
    # chap3 = Seaborn_Intermediate()
    # chap3.exercise_1()
    # chap3.exercise_2()
    # chap3.exercise_3()
    # chap3.ex_4(chap3.bicycle_df )
    # chap3.ex_5(chap3.daily_show_df)

    chap4 = Seaborn_Intermediate()
    # chap4.ex_6(chap4.college_df)
    # chap4.ex_7(chap4.college_df)
    # chap4.ex_8(chap4.college_df)    
    # chap4.ex_9(chap4.car_insurance_df)   
    # chap4.ex_10(chap4.car_insurance_df)   
    # chap4.ex_11(chap4.car_insurance_df)   
    chap4.ex_12(chap4.car_insurance_df)   


if __name__ == "__main__":
    main()