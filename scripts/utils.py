import pandas as pd
import csv
from surprise import Reader
from surprise import Dataset

def load_dataset(path, cols):
    """
    Args:
        path: path to csv file
        cols: which columns to use, if columns dont have a name use indices of the columns, e.g. [0,1,2]
    Returns: a pandas dataframe with csv files in it
    """
    df = pd.read_csv(path, usecols=cols, names=cols)

    return df

def load_movie_titles():
    """
    loads movie titles.
    Returns: a list of all the movie names.
    """
    df = pd.read_csv("../data/ml-latest-small/movies.csv", usecols=['movieId', 'title'])
    df = df.dropna()
    movie_list = df['title'].to_list()


    real_movie_list = movie_list
    placeholder = 'Fooking Workaround!'
    movie_list = [placeholder, *real_movie_list]
    print('nachher\n')
    print(movie_list[76])


    return movie_list


def get_value_counts(dataset1):
    s = list()
    s.append(dataset1.rating.value_counts())

    return s

def get_movie_title():
    reader = Reader(line_format="movieId title genres", sep=',', skip_lines=1)
    ratings_dataset = Dataset.load_from_file('../data/ml-latest-small/movies.csv', reader=reader)

    print(ratings_dataset)



    """ TEST 
    with open("../data/ml-latest-small/movies.csv") as f:
    list2 = [row.split()[0] for row in f]
    id_to_name = {row.split[0]:row.split[1] for row in f}
    """
    movie_id_to_name = {}
    with open('../data/ml-latest-small/ratings.csv', newline='', encoding='ISO-8859-1') as csvfile:
        movie_reader = csv.reader(csvfile)
        next(movie_reader)
        for row in movie_reader:
            movie_ID = row[0]
            movie_name = row[1]
            movie_id_to_name[movie_ID] = movie_name

    print(movie_id_to_name)

    return movie_id_to_name

load_movie_titles()