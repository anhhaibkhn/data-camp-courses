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
    
    def inner_joine_for_sequels(self):
        ''' This function summarize the exercise 'Do sequels earn more?'
        it returns nothing, currently the data for function was not loaded 
        '''
        # empty frame. Data is not loaded 
        sequels, financials = pd.DataFrame(), pd.DataFrame()

        # Merge sequels and financials on index id
        sequels_fin = sequels.merge(financials, on='id', how='left')

        print(sequels.head(3), '\n', sequels.shape)
        print(financials.head(3), '\n', financials.shape)
        print(sequels_fin.head(3), '\n', sequels_fin.shape)

        # Self merge with suffixes as inner join with left on sequel and right on id
        orig_seq = sequels_fin.merge(sequels_fin, how='inner', left_on='sequel', 
                                    right_on='id', right_index=True,
                                    suffixes=('_org','_seq'))

        # Add calculation to subtract revenue_org from revenue_seq 
        orig_seq['diff'] = orig_seq['revenue_seq'] - orig_seq['revenue_org']
        print(orig_seq.head(3), '\n', orig_seq.shape)


        # Select the title_org, title_seq, and diff 
        titles_diff = orig_seq[['title_org','title_seq','diff']]

        # Print the first rows of the sorted titles_diff
        print(titles_diff.sort_values(by='diff', ascending=False ).head())

        '''
            <script.py> output:
                    title sequel
        id                       
        19995       Avatar    nan
        862      Toy Story    863
        863    Toy Story 2  10193 
        
        (4803, 2)
                budget       revenue
        id                             
        19995   237000000  2.787965e+09
        285     300000000  9.610000e+08
        206647  245000000  8.806746e+08 
        (3229, 2)

                    title sequel       budget       revenue
        id                                                  
        19995       Avatar    nan  237000000.0  2.787965e+09
        862      Toy Story    863   30000000.0  3.735540e+08
        863    Toy Story 2  10193   90000000.0  4.973669e+08 
        (4803, 4)

            sequel                                  title_org sequel_org   budget_org  revenue_org                               title_seq sequel_seq   budget_seq   revenue_seq         diff
        id                                                                                                                                                                                   
        862    863                                  Toy Story        863   30000000.0  373554033.0                             Toy Story 2      10193   90000000.0  4.973669e+08  123812836.0
        863  10193                                Toy Story 2      10193   90000000.0  497366869.0                             Toy Story 3        nan  200000000.0  1.066970e+09  569602834.0
        675    767  Harry Potter and the Order of the Phoenix        767  150000000.0  938212738.0  Harry Potter and the Half-Blood Prince        nan  250000000.0  9.339592e+08   -4253541.0 
        (90, 10)

                    title_org        title_seq          diff
        id                                                     
        331    Jurassic Park III   Jurassic World  1.144748e+09
        272        Batman Begins  The Dark Knight  6.303398e+08
        10138         Iron Man 2       Iron Man 3  5.915067e+08
        863          Toy Story 2      Toy Story 3  5.696028e+08
        10764  Quantum of Solace          Skyfall  5.224703e+08
        '''



# def __main__(self):
chap2 = Joining('tmdb-movies.csv')
# print(chap2.movies_table.head())
# print(chap2.movies_table.shape)

taglines = chap2.create_tagline(chap2.df_movies, ['id','tagline'])
# print(taglines.head())
# print(taglines.shape)

tv_genre = chap2.other_join(chap2.movie_to_genres)
