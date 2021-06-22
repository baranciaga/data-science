import pandas as pd
from surprise import Reader
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import random
import matplotlib.pyplot as plt


n = 7824482  #number of records in file
s = 200000 #desired sample size
skip = sorted(random.sample(range(n), n-s))
# skiprows=skip


df = pd.read_csv('../data/ratings_amazon.csv', header=None, usecols=[0, 1, 2],
                 names="user_id,item_id,rating".split(","), nrows = 100000)
reader = Reader(rating_scale=(1, 5))
print(df.shape)

print(df)

df = df.pivot(index='user_id', columns='item_id', values='rating')
print(df)
df.dropna(thresh=20, axis=1, inplace=True)
df = df.stack().reset_index().sort_values(by=['user_id', 'item_id'], axis=0)
df.columns = ['user_id', 'item_id', 'rating']

df.user_id = df.user_id.astype('category').cat.codes.values
df.item_id = df.item_id.astype('category').cat.codes.values

train, test = train_test_split(df, test_size=.2)
n_users, n_items = len(df.user_id.unique()), len(df.item_id.unique())
print(n_users, "\t", n_items)
n_latent_factors = 15

movie_input = keras.layers.Input(shape=[1], name='Item')
# embedding for the movies, see https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9
movie_embedding = keras.layers.Embedding(n_items + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
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


"""item_input = Input(shape=[1], name='Item')
item_embedding = keras.layers.Embedding(n_items+1, n_latent_factors, name='Item-Embedding')(item_input)
item_vec = Flatten(name='FlattenItems')(item_embedding)

user_input = Input(shape=[1], name='User')
user_vec = Flatten(name='FlattenUsers')(
    keras.layers.Embedding(n_users+1, n_latent_factors, name='User-Embedding')(user_input))

prod = keras.layers.dot([item_vec, user_vec], axes=1,  name='DotProduct')
model = keras.Model([user_input, item_input], prod)"""

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
# print(model.summary())

history = model.fit([train.user_id, train.item_id], train.rating, epochs=12, verbose=1, validation_data=([test.user_id, test.item_id], test.rating))
results = model.evaluate((test.user_id, test.item_id), test.rating, batch_size=1)

model.save("../data/model_data/ml/my_model.h5")
np.save('../data/model_data/ml/history.npy', history.history)

pd.Series(history.history['mae']).plot(logy=True)
pd.Series(history.history['val_mae']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.show()
