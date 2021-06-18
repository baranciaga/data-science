from sklearn.datasets import dump_svmlight_file
import numpy as np
import pandas as pd
import os
import urllib
import zipfile
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
import warnings
from surprise import Reader
# TF Modules
import tensorflow as tf
from tensorflow import keras
# from keras.optimizers import Adam
from utils import load_dataset
warnings.filterwarnings('ignore')


# die hier habe ich selbst noch nicht so gecheckt
def recommend(movie_embedding_learnt, user_embedding_learnt, user_id, number_of_movies=5,):
    """
    :param user_embedding_learnt
    :param movie_embedding_learnt
    :param user_id: id of the user
    :param number_of_movies: Number of the movies, default = 5
    :return: an array of movie IDs recommended for the user
    """
    movies = user_embedding_learnt[user_id]@movie_embedding_learnt.T
    mids = np.argpartition(movies, -number_of_movies)[-number_of_movies:]
    return mids


def run_neural_network(df):
    """
    Args: the dataframe to train the model
        df:

    Returns:
        history, model and results

    """

    # load the dataset as a pandas dataframe -> get information out of df.
    dataset1 = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=['userId', 'movieId', 'rating'])
    # dataset1 = load_dataset()
    dataset1.userId = dataset1.userId.astype('category').cat.codes.values
    dataset1.movieId = dataset1.movieId.astype('category').cat.codes.values

    train, test = train_test_split(dataset1, test_size=0.2)

    # get number of unique users and movies, for statistic purposes.
    n_users, n_movies = len(dataset1.userId.unique()), len(dataset1.movieId.unique())
    print(n_users, '\t', n_movies)
    # number of features. In this case genres, appearing actors etc.
    n_latent_factors = 5

    # The model: First layer is an input layer with
    movie_input = keras.layers.Input(shape=[1], name='Item')
    # embedding for the movies, see https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9
    movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

    # same for user
    user_input = keras.layers.Input(shape=[1], name='User')
    user_vec = keras.layers.Flatten(name='FlattenUsers')(
        keras.layers.Embedding(n_users + 1, n_latent_factors, name='User-Embedding')(user_input))

    # dot product
    prod = keras.layers.dot([movie_vec, user_vec], axes=1, name='DotProduct')
    model = keras.Model([user_input, movie_input], prod)

    # compiling the model, specifying the optimizer and the loss function, adam is best for
    # recommender systems usually. See https://ruder.io/optimizing-gradient-descent/
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    print(model.summary())



    # training of the model
    history = model.fit([train.userId, train.movieId], train.rating, epochs=20, verbose=1)
    results = model.evaluate((test.userId, test.movieId), test.rating, batch_size=32)

    movie_embedding_learnt = model.get_layer(name='Movie-Embedding').get_weights()[0]
    print(pd.DataFrame(movie_embedding_learnt).describe())

    user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]
    print(pd.DataFrame(user_embedding_learnt).describe())
    print(history.history.keys())
    print(history.history['loss'])

    pd.Series(history.history['loss']).plot(logy=True)
    plt.xlabel("Epoch")
    plt.ylabel("Training Error")
    plt.show()

    #
    recommend(movie_embedding_learnt, user_embedding_learnt, user_id=1)

    # print(model.metrics_names)

    return history, results, model


run_neural_network(df=1)
