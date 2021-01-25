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
        self.data_by_artist = pd.read_csv('music_data_csv\data_by_artist.csv')