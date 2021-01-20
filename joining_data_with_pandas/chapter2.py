''' Take your knowledge of joins to the next level. In this chapter, you’ll work with TMDb movie data 
as you learn about left, right, and outer joins. 
You’ll also discover how to merge a table to itself and merge on a DataFrame index　'''
import pandas as pd 

class Joining():
    ''' Class allow to using different data joining techniques of pandas'''
    def __init__(self, path):
        self.df_movies = pd.read_csv(path)
        # the following datafram has no data to load
        self.movies = pd.DataFrame()
        self.financials = pd.DataFrame()


    def left_joint(self, df_a, df_b, on_col):
        ''' left join returns only rows of the left_table, and only rows those in the right table 
         A left join will do that by returning all of the rows of your left table, 
         while using an inner join may result in lost data if it does not exist in both tables.
         '''
        df_ab = df_a.merge(df_b, on= on_col, how= 'left')
        return df_ab

    def count_missing_rows_left(self, movies, financials):
        ''' Counting missing rows with left join '''
        # Merge the movies table with the financials table with a left join
        movies_financials = movies.merge(financials, on='id', how='left')

        # Count the number of rows in the budget column that are missing
        number_of_missing_fin = movies_financials['budget'].isnull().sum()

        # Print the number of movies missing financials
        print(number_of_missing_fin)
    


chap2 = Joining('tmdb-movies.csv')

print(chap2.df_movies.head())
print(chap2.df_movies.shape)