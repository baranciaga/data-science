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

# TF Modules

import tensorflow as tf
from tensorflow import keras
# from keras.optimizers import Adam

warnings.filterwarnings('ignore')


def recommend(user_id, number_of_movies=5):
    """
    :param user_id: id of the user
    :param number_of_movies: Number of the movies, default = 5
    :return: an array of movie IDs recommended for the user
    """
    movies = user_embedding_learnt[user_id]@movie_embedding_learnt.T
    mids = np.argpartition(movies, -number_of_movies)[-number_of_movies:]
    return mids


"""
os.chdir('c:/users/Baran/PycharmProjects/data-science/data')
datasets = {'ml100k':'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'ml20m':'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
            'mllatestsmall':'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'ml10m':'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
            'ml1m':'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
            }
dt = 'ml100k'
dt_name = os.path.basename(datasets[dt])

print('Downloading {}'.format(dt_name))
with urllib.request.urlopen(datasets[dt]) as response, open('c:/Users/Baran/PycharmProjects/data-science/data/ml.100k' + dt_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)
print('Download completed')
"""
print(os.getcwd())
dataset = pd.read_csv("../data/ml-100k/u.data",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
train0, test0 = train_test_split(dataset, test_size=.2)
n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
print('++++')
print(n_users, '\t', n_movies)
print('++++')

dataset1=pd.read_csv("../data/ml-latest-small/ratings.csv", usecols = ['userId','movieId', 'rating'])
dataset1.userId = dataset1.userId.astype('category').cat.codes.values
dataset1.movieId = dataset1.movieId.astype('category').cat.codes.values

train, test = train_test_split(dataset1, test_size=0.2)

n_users, n_movies = len(dataset1.userId.unique()), len(dataset1.movieId.unique())
print(n_users, '\t', n_movies)
n_latent_factors = 5

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

history = model.fit([train.userId, train.movieId], train.rating, epochs=1, verbose=1)
results = model.evaluate((test.userId, test.movieId), test.rating, batch_size=1)

movie_embedding_learnt = model.get_layer(name='Movie-Embedding').get_weights()[0]
print(pd.DataFrame(movie_embedding_learnt).describe())

user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]

print("Recommendation: \n")
for i in range(n_users):
    print(recommend(user_id=i))
