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
    dataset1 = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=['userId', 'movieId', 'rating'])
    # dataset1 = load_dataset()
    dataset1.userId = dataset1.userId.astype('category').cat.codes.values
    dataset1.movieId = dataset1.movieId.astype('category').cat.codes.values

    train, test = train_test_split(dataset1, test_size=0.2)

    n_users, n_movies = len(dataset1.userId.unique()), len(dataset1.movieId.unique())
    print(n_users, '\t', n_movies)
    n_latent_factors = 20

    movie_input = keras.layers.Input(shape=[1], name='Item')
    movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

    user_input = keras.layers.Input(shape=[1], name='User')
    user_vec = keras.layers.Flatten(name='FlattenUsers')(
        keras.layers.Embedding(n_users + 1, n_latent_factors, name='User-Embedding')(user_input))

    prod = keras.layers.dot([movie_vec, user_vec], axes=1, name='DotProduct')
    model = keras.Model([user_input, movie_input], prod)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    print(model.summary())

    history = model.fit([train.userId, train.movieId], train.rating, epochs=5, verbose=1)
    results = model.evaluate((test.userId, test.movieId), test.rating, batch_size=32)

    movie_embedding_learnt = model.get_layer(name='Movie-Embedding').get_weights()[0]
    print(pd.DataFrame(movie_embedding_learnt).describe())

    user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]
    print(pd.DataFrame(user_embedding_learnt).describe())
    print(history.history.keys())
    print(history.history['loss'])

    recommend(movie_embedding_learnt, user_embedding_learnt, user_id=1)

    # print(model.metrics_names)

    return history, results, model


run_neural_network(df=1)
