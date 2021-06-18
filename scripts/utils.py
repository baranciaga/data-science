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
    loads movie titles.
    Returns: a list of all the movie names.
    """
    df = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=['movieId', 'title', 'genres'])
    df = df.dropna()
    movie_list = df['title'].to_list()

    return movie_list


def get_value_counts(dataset1):
    s = list()
    s.append(dataset1.rating.value_counts())

    return s
