import streamlit as st
import yfinance as yf
from knn_movie_rec import apply_kNN_movie, get_top_n
import pandas as pd
import numpy as np




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
        # choice_dataset = st.selectbox('Please select the dataset:', ('Amazon reviews', 'Movielens movie reviews'))
        # number = st.slider("Popularity Threshold", 1, 5, )
        threshold = st.slider('Threshold: ', 1, 5)
        choice_of_k = st.slider("Choose k", 1, 10)
        choice_of_n = st.slider("How many recommendations do you wanna get?", 1, 10)
        choice_of_algo = st.selectbox("Which Similarity metric?", ("pearson", "pearson_baseline", "msd", "cosine"))

        if st.button('Run Algo'):
            apply_kNN_movie(int(threshold), str(choice_of_algo), int(choice_of_k), int(choice_of_n))

    if page_selection == "Neural network matrix factorization":
        print("hello")


if __name__ == '__main__':
    main()
