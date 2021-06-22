import collections
from collections import defaultdict

import numpy as np
import pandas as pd
from surprise import Dataset,KNNBaseline, accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

# Daten in ein pandas dataframe einlesen, file liegt im Projektordner
df = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols = ['userId','movieId', 'rating'])
print(df)
print(df["rating"].mean())
#Tabelle in user-item matrix umwandeln. rows = users, columns = items
df = df.pivot(index = 'userId', columns ='movieId', values = 'rating')
print(df)
#Anzahl an NaN ratings im gesamten df
print(df.isnull().sum().sum())
#Anzahl ratings im gesamten df
print(np.sum(df.count()))

'''
#Tabelle in user-item matrix umwandeln. rows = users, columns = items
df = df.pivot(index = 'userId', columns ='movieId', values = 'rating')
print(df)
# columns rauslöschen, wo Anzahl ratings < thresh
df.dropna(thresh=40 ,axis=1, inplace=True)
print(df)
# Matrix wieder in Tabellenform umwandeln. Table: userId, movieId, ratings sortiert nach userId
df = df.stack().reset_index().sort_values(by=['userId', 'movieId'], axis=0)
df.columns = ['userId','movieId', 'rating']
print(df)
'''