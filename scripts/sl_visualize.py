import streamlit as st
from knn_movie_rec import apply_kNN_movie, get_top_n
import pandas as pd
from utils import load_dataset



def load_movie_titles():
    """
    loads movie titles.
    Returns: a list of all the movie names.
    """
    df = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=['movieId', 'title', 'genres'])
    df = df.dropna()
    movie_list = df['title'].to_list()

    return movie_list


def main():
    """
    main method of the streamlit UI
    Returns: None

    """
    st.write("""
    # Barans and Dungs Recommender
    """)
    page_options = ["K-nearest neighbor", "Neural network matrix factorization"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "K-nearest neighbor":
        choice_dataset = st.selectbox('Please select the dataset:', ('Amazon reviews', 'Movielens movie reviews'))

        if choice_dataset == 'Amazon reviews':
            path_to_dataset = "../data/ratings_amazon.csv"
            cols = [0, 1, 2]

        elif choice_dataset == 'Movielens movie reviews':
            path_to_dataset = "../data/ml-latest-small/ratings.csv"
            cols = ['userId', 'movieId', 'rating']

        df = load_dataset(path=path_to_dataset, cols=cols)
        st.write(df.shape)
        st.write(df.head(20))
        # number = st.slider("Popularity Threshold", 1, 5, )
        threshold = st.slider('Threshold: ', 1, 5)
        choice_of_k = st.slider("Choose k", 1, 10)
        choice_of_n = st.slider("How many recommendations do you wanna get?", 1, 10)
        choice_of_algo = st.selectbox("Which Similarity metric?", ("pearson", "pearson_baseline", "msd", "cosine"))

        if st.button('Run KNN Algo'):
            result, error = apply_kNN_movie(int(threshold), str(choice_of_algo), int(choice_of_k), int(choice_of_n))
            get_top_n(result)

    if page_selection == "Neural network matrix factorization":
        choice_dataset = st.selectbox('Please select the dataset:', ('Amazon reviews', 'Movielens movie reviews'))

        if choice_dataset == 'Amazon reviews':
            path_to_dataset = "../data/ratings_amazon.csv"
            cols = [0, 1, 2]
        elif choice_dataset == 'Movielens movie reviews':
            path_to_dataset = "../data/ml-latest-small/ratings.csv"
            cols = ['userId', 'movieId', 'rating']

        df = load_dataset(path=path_to_dataset, cols=cols)
        st.write(df.head(20))
        n_epochs = st.slider('Epochs', 1, 20)
        choice_metrics = st.selectbox('Which metrics?', ('mae', 'mse'))
        choice_optimizer = st.selectbox('Optimizer', 'adam')


if __name__ == '__main__':
    main()
