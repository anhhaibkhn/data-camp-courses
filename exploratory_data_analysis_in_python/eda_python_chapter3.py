''' Chapter 3: Relationships
Up until this point, you've only looked at one variable at a time. 
In this chapter, you'll explore relationships between variables two at a time, 
using scatter plots and other visualizations to extract insights from a new dataset obtained from 
the Behavioral Risk Factor Surveillance Survey (BRFSS). 
You'll also learn how to quantify those relationships using correlation and simple regression.
'''
from eda_python_chapter2 import Distributions


class Relationships():
    ''' class to check the data distribution'''

    def __init__ (self):
        # initialize the Distribution object
        self.chap2_dist = Distributions()


    def new_func(self):
        print('begin chap3')
        print(self.chap2_dist.gss.head())




# main
chap3 = Relationships()


## Exploring relationships
chap3.new_func()
