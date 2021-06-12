import pandas as pd
from surprise import Dataset
from surprise import Reader
# from surprise.model_selection import train_test_split
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from keras.layers import Input, Embedding, Flatten, Dot
from keras.models import Model

print(os.getcwd())
df = pd.read_csv('../data/ratings_amazon.csv', header=None, usecols=[0, 1, 2],
                 names="user_id,item_id,rating".split(","))
reader = Reader(rating_scale=(1, 5))
print('!!!')
print(df.head())
print('!!!')
"""
data = Dataset.load_from_df(df, reader)
x_train, x_test = train_test_split(data, train_size = .8)
print(len(x_test))
"""
df.user_id = df.user_id.astype('category').cat.codes.values
df.item_id = df.item_id.astype('category').cat.codes.values

train, test = train_test_split(df, test_size=.2)
n_users, n_items = len(df.user_id.unique()), len(df.item_id.unique())
print(n_users, "\t", n_items)
n_latent_factors = 20


item_input = Input(shape=[1], name='Item')
item_embedding = Embedding(n_items+1, n_latent_factors, name='item_embedding')(item_input)
item_vec = Flatten(name='FlattenItems')(item_embedding)

user_input = Input(shape=[1], name='User')
user_vec = Flatten(name='FlattenUsers')(Embedding(n_users+1, n_latent_factors, name='user_embedding')(user_input))

# prod = Dot([item_vec, user_vec], axes=1,  name='dot_product')
#model = Model([user_input, item_input], prod)

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
# print(model.summary())
print('EOF')