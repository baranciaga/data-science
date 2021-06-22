import pandas as pd


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
    *** USE dict.get(n) to get the movie ID n and the title
    loads movie titles.
    Returns: a list of all the movie names.
    """
    df = pd.read_csv("../data/ml-latest-small/movies.csv", usecols=['movieId', 'title'])
    df = df.dropna()
    movie_dict = dict(zip(df['movieId'], df['title']))

    return movie_dict


def get_value_counts(dataset1):
    """

    :param dataset1: the source dataset
    :return: value counts of the rating column
    """
    s = list()
    s.append(dataset1.rating.value_counts())

    return s


load_movie_titles()
