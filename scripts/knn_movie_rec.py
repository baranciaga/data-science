import pandas as pd
from surprise import Dataset, KNNBasic, KNNBaseline, KNNWithZScore, SVD, accuracy
from surprise import Reader
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split

# Daten in ein pandas dataframe einlesen, file liegt im Projektordner
df = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols = ['userId','movieId', 'rating'])
print(df)

# pandas dataframe in ein Surprise data object umwandeln. nur relevante spalten auswählen
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

# test und trainset erstellen
x_train, x_test= train_test_split(data, train_size=0.5, test_size= 0.5)
print("train data \n", x_train)
#print("test data \n", x_test)

# To use item-based cosine similarity
sim_options = {
    "name": "pearson", #oder cosine, oder pearson_baseline
    "user_based": True,  # Compute  similarities between items
}
# definition des algo objekts
algo1 = KNNWithMeans(k=2, sim_options=sim_options)

# algo trainieren
algo1.fit(x_train)

# algo testen
predictions1 = algo1.test(x_test)

# Evaluations berechnen
accuracy.rmse(predictions1)
accuracy.mae(predictions1)
accuracy.mse(predictions1)