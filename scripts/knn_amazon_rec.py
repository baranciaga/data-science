import collections
from collections import defaultdict
import pandas as pd
from surprise import Dataset,KNNWithMeans, accuracy
from surprise import Reader
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
    #sort by User
    top_n = collections.OrderedDict(sorted(top_n.items()))

    return top_n

def apply_kNN_amazon (threshold, similarity_metric, user_based, k, n ):

    # Daten in ein pandas dataframe einlesen, file liegt im Projektordner, header muss erstellt werden weil keine im original file sind
    df = pd.read_csv('../data/ratings_amazon.csv',header = None , usecols = [0,1,2], names=['userId','productId', 'rating'], nrows = 100000)
    # Tabelle in user-item matrix umwandeln. rows = users, columns = items
    df = df.pivot(index='userId', columns='productId', values='rating')
    # columns rauslöschen, wo Anzahl ratings < thresh
    df.dropna(thresh=threshold, axis=1, inplace=True)
    # Matrix wieder in Tabellenform umwandeln. Table: userId, movieId, ratings sortiert nach userId
    df = df.stack().reset_index().sort_values(by=['userId', 'productId'], axis=0)
    df.columns = ['userId', 'productId', 'rating']
    # pandas dataframe in ein Surprise data object umwandeln
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userId", "productId", "rating"]], reader)

    # test und trainset erstellen
    x_train, x_test= train_test_split(data, train_size=0.8, test_size= 0.2)

    # To use item-based pearson similarity
    sim_options = {
        "name": "pearson",
        "user_based": True,  # Compute  similarities between items
    }
    # definition des algo objekts
    algo1 = KNNWithMeans(k=2, sim_options=sim_options)

    # algo trainieren
    algo1.fit(x_train)
    # algo testen
    predictions1 = algo1.test(x_test)
    # für jeden user die n items, wo wir predicten dass er sie hoch bewertet, holen
    top_n = get_top_n(predictions1, n=n)
    # Evaluations berechnen
    error_score = accuracy.mae(predictions1)

    return top_n, error_score

'''CF Anwenden mit params. Returnt top_n (die top predicteten ratings zu jeden user) und den error score'''
#params: threshold, similarity, user_based, k, n
top_n, error_score = apply_kNN_movie(100, "pearson", False, 1000, 5)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

print(error_score)

print(top_n[610])
