''' Take your knowledge of joins to the next level. In this chapter, you’ll work with TMDb movie data 
as you learn about left, right, and outer joins. 
You’ll also discover how to merge a table to itself and merge on a DataFrame index　'''
import pandas as pd
import numpy as np

class Joining():
    ''' Class allow to using different data joining techniques of pandas'''
    def __init__(self, path):
        self.df_movies = pd.read_csv(path)
        # the following datafram has no data to load
        self.movies = pd.DataFrame()
        self.financials = pd.DataFrame()

        # create tmdb_movies.csv with shape(4083,4)
        self.movies_table = self.df_movies[['id', 'original_title', 'popularity', 'release_date']][:4084]    
        self.movies_table.to_csv('tmdb_movies.csv')

        # create tmdb_movie_to_genres.csv
        self.tmdb_5000_movies = pd.read_csv('archive/tmdb_5000_movies.csv')
        self.movie_to_genres = self.tmdb_5000_movies[['movie_id', 'genre']]
        self.movie_to_genres.to_csv('tmdb_movie_to_genres.csv')


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
    
    def create_tagline(self, df, arr):
        """ create tmdb_taglines.csv datset for joining practice 
            shape (3955,3)
        """
        taglines = df[arr][:3956]
        taglines.to_csv('tmdb_taglines.csv')

        return taglines
    
    def other_join(self, movie_to_genres):

        tv_genre = movie_to_genres[movie_to_genres['genre'] == 'TV Movie']
        print(tv_genre.head(10))

        return tv_genre

# def __main__(self):
chap2 = Joining('tmdb-movies.csv')
# print(chap2.movies_table.head())
# print(chap2.movies_table.shape)

taglines = chap2.create_tagline(chap2.df_movies, ['id','tagline'])
# print(taglines.head())
# print(taglines.shape)

tv_genre = chap2.other_join(chap2.movie_to_genres)
