'''
Chapter 3: Advanced Merging and Concatenating

In this chapter, you’ll leverage powerful filtering techniques, including semi-joins and anti-joins. 
You’ll also learn how to glue DataFrames by vertically combining and using the pandas.concat function 
to create new datasets. Finally, because data is rarely clean,
you’ll also learn how to validate your newly combined data structures.
'''
import pandas as pd
import numpy as np

class Advance_Merge_Concatenate():
    ''' Class allow to using different data joining techniques of pandas'''
    def __init__(self):
        # the following datafram has no data to load
        self.data_by_artist = pd.read_csv('music_data_csv\data_by_artist.csv')
        self.data_by_genres = pd.read_csv('music_data_csv\data_by_genres.csv')
        self.data_by_year = pd.read_csv('music_data_csv\data_by_year.csv')
        self.data_w_genres = pd.read_csv('music_data_csv\data_w_genres.csv')

    # the following fucntions are only for code exercises memo, dataset was not collected for practice.
    def anti_join(self):
        ''' anti-joins'''
        # anti join
        employees, top_cust = pd.DataFrame() , pd.DataFrame()

        print(employees.head(3), '\n', employees.shape) 
        print(top_cust.head(3), '\n', top_cust.shape) 
        # Merge employees and top_cust
        empl_cust = employees.merge(top_cust, on='srid', how='left', indicator=True)
        print(empl_cust.head(3), '\n', empl_cust.shape)

        # Select the srid column where _merge is left_only
        srid_list = empl_cust.loc[empl_cust['_merge'] == 'left_only', 'srid']

        # Get employees not working with top customers
        print(employees[employees['srid'].isin(srid_list)])
        # semi join
        '''<script.py> output:
        srid    lname   fname                title  hire_date                   email
        0     1    Adams  Andrew      General Manager 2002-08-14  andrew@chinookcorp.com
        1     2  Edwards   Nancy        Sales Manager 2002-05-01   nancy@chinookcorp.com
        2     3  Peacock    Jane  Sales Support Agent 2002-04-01    jane@chinookcorp.com 
        (8, 6)
        cid  srid     fname      lname               phone                 fax                  email
        0    1     3      Luís  Gonçalves  +55 (12) 3923-5555  +55 (12) 3923-5566   luisg@embraer.com.br
        1    2     5    Leonie     Köhler    +49 0711 2842222                 NaN  leonekohler@surfeu.de
        2    3     3  François   Tremblay   +1 (514) 721-4711                 NaN    ftremblay@gmail.com 
        (59, 7)
        srid  lname_x fname_x                title  hire_date  ...    lname_y               phone                 fax               email_y     _merge
        0     1    Adams  Andrew      General Manager 2002-08-14  ...        NaN                 NaN                 NaN                   NaN  left_only
        1     2  Edwards   Nancy        Sales Manager 2002-05-01  ...        NaN                 NaN                 NaN                   NaN  left_only
        2     3  Peacock    Jane  Sales Support Agent 2002-04-01  ...  Gonçalves  +55 (12) 3923-5555  +55 (12) 3923-5566  luisg@embraer.com.br       both
        
        [3 rows x 13 columns] 
        (64, 13)
        srid     lname    fname            title  hire_date                    email
        0     1     Adams   Andrew  General Manager 2002-08-14   andrew@chinookcorp.com
        1     2   Edwards    Nancy    Sales Manager 2002-05-01    nancy@chinookcorp.com
        5     6  Mitchell  Michael       IT Manager 2003-10-17  michael@chinookcorp.com
        6     7      King   Robert         IT Staff 2004-01-02   robert@chinookcorp.com
        7     8  Callahan    Laura         IT Staff 2004-03-04    laura@chinookcorp.com
        '''


    def semi_join(self):
        '''semi join exercise'''
        non_mus_tcks, top_invoices, genres  = pd.DataFrame() , pd.DataFrame(), pd.DataFrame()

        # Merge the non_mus_tck and non_mus_tck tables on tid
        print(non_mus_tcks.head(3), '\n', non_mus_tcks.shape) 
        print(top_invoices.head(3), '\n', top_invoices.shape) 
        print(genres.head(3), '\n', genres.shape)

        tracks_invoices = non_mus_tcks.merge(top_invoices, on='tid', how='inner')
        print(tracks_invoices.head(3), '\n', tracks_invoices.shape)

        # Use .isin() to subset non_mus_tcks to rows with tid in tracks_invoices
        top_tracks = non_mus_tcks[non_mus_tcks['tid'].isin(tracks_invoices['tid'])]


        # Group the top_tracks by gid and count the tid rows
        cnt_by_gid = top_tracks.groupby(['gid'], as_index=False).agg({'tid':'count' })

        # Merge the genres table to cnt_by_gid on gid and print
        print(cnt_by_gid.merge(genres, on='gid'))

        '''<script.py> output:
            tid                    name  aid  mtid  gid  u_price
        2819  2820  Occupation / Precipice  227     3   19     1.99
        2820  2821           Exodus, Pt. 1  227     3   19     1.99
        2821  2822           Exodus, Pt. 2  227     3   19     1.99 
        (200, 6)
            ilid  iid   tid  uprice  quantity
        469   470   88  2832    1.99         1
        472   473   88  2850    1.99         1
        475   476   88  2868    1.99         1 
        (16, 5)
        gid   name
        0    1   Rock
        1    2   Jazz
        2    3  Metal 
        (25, 2)
            tid       name  aid  mtid  gid  u_price  ilid  iid  uprice  quantity
        0  2850    The Fix  228     3   21     1.99   473   88    1.99         1
        1  2850    The Fix  228     3   21     1.99  2192  404    1.99         1
        2  2868  Walkabout  230     3   19     1.99   476   88    1.99         1 
        (14, 10)
        gid  tid      name
        0   19    4  TV Shows
        1   21    2     Drama
        2   22    1    Comedy
        '''

    def concat_basic(self, tracks_master, tracks_ride,  tracks_st):
        '''Concatenate tracks_master, tracks_ride, and tracks_st, in that order, setting sort to True. 
        '''
        # Concatenate the tracks
        tracks_from_albums = pd.concat([tracks_master, tracks_ride,  tracks_st],sort=True)
        print(tracks_from_albums)
        '''Concatenate tracks_master, tracks_ride, and tracks_st, where the index goes from 0 to n-1. '''
        # Concatenate the tracks so the index goes from 0 to n-1
        tracks_from_albums = pd.concat([tracks_master, tracks_ride,  tracks_st],
                                    ignore_index=True, 
                                    sort=True)
        print(tracks_from_albums)
        '''Concatenate tracks_master, tracks_ride, and tracks_st, showing only columns that are in all tables.'''
        # Concatenate the tracks, show only columns names that are in all tables
        tracks_from_albums = pd.concat([tracks_master, tracks_ride,  tracks_st],
                                    join='inner',
                                    sort=True)
        print(tracks_from_albums)


    def concat_with_key(self, inv_jul, inv_aug, inv_sep):
        '''Concatenate the three tables together vertically in order with the oldest month first, 
        adding '7Jul', '8Aug', and '9Sep' as keys for their respective months, and save to variable avg_inv_by_month.
        Use the .agg() method to find the average of the total column from the grouped invoices.
        Create a bar chart of avg_inv_by_month.'''
        # Concatenate the tables and add keys
        inv_jul_thr_sep = pd.concat([inv_jul, inv_aug, inv_sep], 
                                    keys=['7Jul', '8Aug',  '9Sep'])

        # Group the invoices by the index keys and find avg of the total column
        avg_inv_by_month = inv_jul_thr_sep.groupby(level=0).agg({'total':'mean'})

        # Bar plot of avg_inv_by_month
        avg_inv_by_month.plot(kind='bar')
        plt.show()

    def using_append(self,tracks_ride, tracks_master, tracks_st, invoice_items):
        '''The .concat() method is excellent when you need a lot of control over how concatenation is performed.
         However, if you do not need as much control, then the .append() method is another option. 
         You'll try this method out by appending the track lists together from different Metallica albums. 
        From there, you will merge it with the invoice_items table to determine which track sold the most.
        '''
        # Use the .append() method to combine the tracks tables
        metallica_tracks = tracks_ride.append([tracks_master, tracks_st], sort=False)

        # Merge metallica_tracks and invoice_items
        tracks_invoices = metallica_tracks.merge(invoice_items, on='tid', how='inner')

        # For each tid and name sum the quantity sold
        tracks_sold = tracks_invoices.groupby(['tid','name']).agg({'quantity':'sum'})

        # Sort in decending order by quantity and print the results
        print(tracks_sold.sort_values(by='quantity', ascending=False))
        '''
        tracks_invoices

            tid                     name  aid  mtid  gid  ...             composer  ilid  iid  uprice  quantity
        0   1875       Ride The Lightning  154     1    3  ...                  NaN   887  165    0.99         1
        1   1876  For Whom The Bell Tolls  154     1    3  ...                  NaN   312   59    0.99         1
        2   1876  For Whom The Bell Tolls  154     1    3  ...                  NaN  1461  270    0.99         1
        3   1877            Fade To Black  154     1    3  ...                  NaN  2035  375    0.99         1
        4   1853                  Battery  152     1    3  ...  J.Hetfield/L.Ulrich   882  164    0.99         1
        5   1853                  Battery  152     1    3  ...  J.Hetfield/L.Ulrich  2031  375    0.99         1
        6   1854        Master Of Puppets  152     1    3  ...            K.Hammett   302   55    0.99         1
        7   1857        Disposable Heroes  152     1    3  ...  J.Hetfield/L.Ulrich   883  164    0.99         1
        8   1882                  Frantic  155     1    3  ...                  NaN  1462  270    0.99         1
        9   1884     Some Kind Of Monster  155     1    3  ...                  NaN   314   59    0.99         1
        10  1886            Invisible Kid  155     1    3  ...                  NaN  2036  376    0.99         1

        [11 rows x 11 columns]

        <script.py> output:
                                        quantity
            tid  name                             
            1853 Battery                         2
            1876 For Whom The Bell Tolls         2
            1854 Master Of Puppets               1
            1857 Disposable Heroes               1
            1875 Ride The Lightning              1
            1877 Fade To Black                   1
            1882 Frantic                         1
            1884 Some Kind Of Monster            1
            1886 Invisible Kid                   1
        '''









chap3 = Advance_Merge_Concatenate()
print(chap3.data_by_artist.head(), '\n', chap3.data_by_artist.shape)
print(chap3.data_by_genres.head(), '\n', chap3.data_by_genres.shape)
print(chap3.data_by_year.head(), '\n', chap3.data_by_year.shape)