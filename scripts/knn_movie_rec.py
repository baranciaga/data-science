from collections import defaultdict

import pandas as pd
from surprise import Dataset, KNNBasic, KNNBaseline, KNNWithZScore, SVD, accuracy
from surprise import Reader
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split

# von https://surprise.readthedocs.io/en/stable/FAQ.html
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

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

# für jeden user die 5 items, wo wir predicten dass er sie hoch bewertet, holen
top_n = get_top_n(predictions1, n=5)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])