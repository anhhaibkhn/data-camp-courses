'''
Chapter 4: Merging Ordered and Time-Series Data

In this final chapter, you’ll step up a gear and learn to apply pandas' specialized methods 
for merging time-series and ordered data together with real-world financial and economic data from the city of Chicago. 
You’ll also learn how to query resulting tables using a SQL-style format, and unpivot data using the melt method.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Merging_Ordered_for_Time_Series_Data():
    ''' Class allow to  gear and learn to apply pandas methods for merging time-series and ordered data together.
        It also included pandas query methods, sql styles, unpivot-data, melt methods. 
    '''
    def __init__(self):
        # the following datafram has no data to load
        self.sp500 = pd.DataFrame()
    
    def compute_the_gdp_correlation(self, sp500, gdp):
        ''' Merge the different datasets together to compute the correlation between GDP and S&P500'''
        # Use merge_ordered() to merge gdp and sp500, interpolate missing value
        gdp_sp500 = pd.merge_ordered(gdp, sp500, left_on='year', right_on='date', 
                                    how='left',  fill_method='ffill')

        # Subset the gdp and returns columns
        gdp_returns = gdp_sp500[['gdp', 'returns']]

        # Print gdp_returns correlation
        print(gdp_returns.corr())
        ''' <script.py> output:
                        gdp   returns
            gdp      1.000000  0.212173
            returns  0.212173  1.000000
        '''

    def Phillips_curve_using_merge_ordered(self,unemployment , inflation ):
        # Use merge_ordered() to merge inflation, unemployment with inner join
        print(inflation.head(5), '\n', inflation.shape)
        print(unemployment.head(5), '\n', unemployment.shape)
        '''
        <script.py> output:
                date      cpi     seriesid                  data_type
        0  2014-01-01  235.288  CUSR0000SA0  SEASONALLY ADJUSTED INDEX
        1  2014-02-01  235.547  CUSR0000SA0  SEASONALLY ADJUSTED INDEX
        2  2014-03-01  236.028  CUSR0000SA0  SEASONALLY ADJUSTED INDEX
        3  2014-04-01  236.468  CUSR0000SA0  SEASONALLY ADJUSTED INDEX
        4  2014-05-01  236.918  CUSR0000SA0  SEASONALLY ADJUSTED INDEX 
        (60, 4)

                date  unemployment_rate
        0  2013-06-01                7.5
        1  2014-01-01                6.7
        2  2014-06-01                6.1
        3  2015-01-01                5.6
        4  2015-06-01                5.3 
        (14, 2)
        '''
        inflation_unemploy = pd.merge_ordered(inflation, unemployment, on='date', how='inner', fill_method='ffill')
        '''
                        date      cpi     seriesid                  data_type  unemployment_rate
        0  2014-01-01  235.288  CUSR0000SA0  SEASONALLY ADJUSTED INDEX                6.7
        1  2014-06-01  237.231  CUSR0000SA0  SEASONALLY ADJUSTED INDEX                6.1
        2  2015-01-01  234.718  CUSR0000SA0  SEASONALLY ADJUSTED INDEX                5.6
        3  2015-06-01  237.684  CUSR0000SA0  SEASONALLY ADJUSTED INDEX                5.3
        4  2016-01-01  237.833  CUSR0000SA0  SEASONALLY ADJUSTED INDEX                5.0 
        (10, 5)
        '''
        # Print inflation_unemploy 
        print(inflation_unemploy.head(5), '\n', inflation_unemploy.shape)

        # Plot a scatter plot of unemployment_rate vs cpi of inflation_unemploy
        inflation_unemploy.plot(x='unemployment_rate',y='cpi', kind='scatter')
        plt.show()

    def merge_order_multiple_columns(self, gdp , pop):
        ''' merge_ordered() caution, multiple columns
        we will merge GDP and population data from the World Bank for the Australia and Sweden, 
        reversing the order of the merge on columns. The frequency of the series are different, 
        the GDP values are quarterly, and the population is yearly.
        Use the forward fill feature to fill in the missing data. 
        '''
        print(gdp.head(5), '\n', gdp.shape)
        print(pop.head(5), '\n', pop.shape)
        '''
                date    country          gdp    series_code
        0 1990-01-01  Australia  158051.1324  NYGDPMKTPSAKD
        1 1990-04-01  Australia  158263.5816  NYGDPMKTPSAKD
        2 1990-07-01  Australia  157329.2790  NYGDPMKTPSAKD
        3 1990-09-01  Australia  158240.6781  NYGDPMKTPSAKD
        4 1991-01-01  Australia  156195.9535  NYGDPMKTPSAKD 
        (32, 4)
                date    country       pop  series_code
        0 1990-01-01  Australia  17065100  SP.POP.TOTL
        1 1991-01-01  Australia  17284000  SP.POP.TOTL
        2 1992-01-01  Australia  17495000  SP.POP.TOTL
        3 1993-01-01  Australia  17667000  SP.POP.TOTL
        4 1990-01-01     Sweden   8558835  SP.POP.TOTL 
        '''

        # Merge gdp and pop on country and date with fill
        date_ctry = pd.merge_ordered(gdp , pop, on=['country', 'date'],
                                    fill_method='ffill')

        # Print date_ctry
        print(date_ctry.head(5), '\n', date_ctry.shape)
        '''
                    date    country          gdp  series_code_x       pop series_code_y
        0 1990-01-01  Australia  158051.1324  NYGDPMKTPSAKD  17065100   SP.POP.TOTL
        1 1990-04-01  Australia  158263.5816  NYGDPMKTPSAKD  17065100   SP.POP.TOTL
        2 1990-07-01  Australia  157329.2790  NYGDPMKTPSAKD  17065100   SP.POP.TOTL
        3 1990-09-01  Australia  158240.6781  NYGDPMKTPSAKD  17065100   SP.POP.TOTL
        4 1991-01-01  Australia  156195.9535  NYGDPMKTPSAKD  17284000   SP.POP.TOTL
         (32, 6)
        '''

        # Merge gdp and pop on date and country with fill and notice rows 2 and 3
        ctry_date = pd.merge_ordered(gdp , pop, on=['date', 'country'],
                                    fill_method='ffill')

        # Print ctry_date
        print(ctry_date.head(5))
        '''
                        date    country           gdp  series_code_x       pop series_code_y
        0 1990-01-01  Australia  158051.13240  NYGDPMKTPSAKD  17065100   SP.POP.TOTL
        1 1990-01-01     Sweden   79837.84599  NYGDPMKTPSAKD   8558835   SP.POP.TOTL
        2 1990-04-01  Australia  158263.58160  NYGDPMKTPSAKD   8558835   SP.POP.TOTL
        3 1990-04-01     Sweden   80582.28597  NYGDPMKTPSAKD   8558835   SP.POP.TOTL
        4 1990-07-01  Australia  157329.27900  NYGDPMKTPSAKD   8558835   SP.POP.TOTL 
        (32, 6)
        '''
        def merge_asof_one(self, jpm, wells, bac):
            ''' 
            Using merge_asof() to study stocks
            '''
            # Use merge_asof() to merge jpm and wells
            jpm_wells = pd.merge_asof(jpm, wells, on='date_time',direction='nearest', suffixes=('', '_wells'))

            # Use merge_asof() to merge jpm_wells and bac
            jpm_wells_bac =  pd.merge_asof(jpm_wells, bac, on='date_time',direction='nearest', suffixes=('_jpm', '_bac'))

            # Compute price diff
            price_diffs = jpm_wells_bac.diff()

            # Plot the price diff of the close of jpm, wells and bac only
            price_diffs.plot(y=['close_jpm', 'close_wells', 'close_bac'])
            plt.show()
        
        def merge_asof_two(self, gdp, recession):
            '''
                Using merge_asof() to create dataset
            '''
             # Merge gdp and recession on date using merge_asof()
            gdp_recession = pd.merge_asof(gdp, recession, on='date')

            # Create a list based on the row value of gdp_recession['econ_status']
            is_recession = ['r' if s=='recession' else 'g' for s in gdp_recession['econ_status']]

            # Plot a bar chart of gdp_recession
            gdp_recession.plot(kind='bar', y='gdp', x='date', color=is_recession, rot=90)
            plt.show()
        
        def pandas_query(self, ):
            ''' 
                Subsetting rows with .query()
            '''
            # Merge gdp and pop on date and country with fill
            gdp_pop = pd.merge_ordered(gdp, pop, on=['country','date'], fill_method='ffill')

            # Add a column named gdp_per_capita to gdp_pop that divides the gdp by pop
            gdp_pop['gdp_per_capita'] = gdp_pop['gdp'] / gdp_pop['pop']

            # Pivot data so gdp_per_capita, where index is date and columns is country
            gdp_pivot = gdp_pop.pivot_table('gdp_per_capita', 'date', 'country')
            print(gdp_pivot, gdp_pivot.shape)
            # Select dates equal to or greater than 1991-01-01
            recent_gdp_pop = gdp_pivot.query('date >= 1991')

            # Plot recent_gdp_pop
            recent_gdp_pop.plot(rot=90)
            plt.show()

        def using_pandas_melt(self, ur_wide):
            '''
            Using .melt() to reshape government data
            '''
            ur_wide.head()
            # unpivot everything besides the year column
            ur_tall = ur_wide.melt(id_vars='year', var_name=['month'], value_name='unempl_rate')
            ur_tall.head()

            # Create a date column using the month and year columns of ur_tall
            ur_tall['date'] = pd.to_datetime(ur_tall['year'] + '-' + ur_tall['month'])

            # Sort ur_tall by date in ascending order
            ur_sorted = ur_tall.sort_values(by='date')
            ur_sorted.head()
            # Plot the unempl_rate by date
            ur_sorted.plot(y='unempl_rate', x='date')
            plt.show()

        def using_pandas_melt_two(self,ten_yr ):
            '''
            Using .melt() for stocks vs bond performance
            '''
            print(ten_yr.head())
            # Use melt on ten_yr, unpivot everything besides the metric column
            bond_perc = ten_yr.melt(id_vars='metric', var_name=['date'], value_name='close' )
            print(bond_perc.head())
            # Use query on bond_perc to select only the rows where metric=close
            bond_perc_close = bond_perc.query('metric == "close"')
            print(bond_perc_close.head())
            # Merge (ordered) dji and bond_perc_close on date with an inner join
            dow_bond = pd.merge_ordered(dji, bond_perc_close, on='date',how ='inner', suffixes=('_dow', '_bond'))
            print(dow_bond.head())

            # Plot only the close_dow and close_bond columns
            dow_bond.plot(y=['close_dow','close_bond'], x='date', rot=90)
            plt.show()

            
                    















