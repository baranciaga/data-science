import streamlit as st
# from knn_movie_rec import apply_kNN_movie, get_top_n
import pandas as pd
from utils import load_dataset
# from nnmf_movie_rec import run_neural_network





def main():
    """
    main method of the streamlit UI
    Returns: None
    """
    # title
    st.write("""
    # Barans and Dungs Recommender
    """)
    # Choice of the algo
    page_options = ["K-nearest neighbor", "Neural network matrix factorization"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "K-nearest neighbor":
        choice_dataset = st.selectbox('Please select the dataset:', ('Amazon reviews', 'Movielens movie reviews'))

        # choice of dataset, this code is duplicated in line 62-67
        if choice_dataset == 'Amazon reviews':
            path_to_dataset = "../data/ratings_amazon.csv"
            cols = [0, 1, 2]

        elif choice_dataset == 'Movielens movie reviews':
            path_to_dataset = "../data/ml-latest-small/ratings.csv"
            cols = ['userId', 'movieId', 'rating']

        df = load_dataset(path=path_to_dataset, cols=cols)

        #output of data
        st.write("Shape of Data", df.shape)
        st.write(df.head(20))
        # input variables for the knn algo:
        # number = st.slider("Popularity Threshold", 1, 5, )
        threshold = st.sidebar.number_input('Threshold: ', value=1)
        choice_of_k = st.sidebar.number_input("Choose k", value=10)
        choice_of_n = st.sidebar.number_input("How many recommendations do you wanna get?", value=10)
        choice_of_algo = st.sidebar.selectbox("Which Similarity metric?", ("pearson", "pearson_baseline", "msd", "cosine"))

        # button for knn algo
        if st.sidebar.button('Run KNN Algo'):
            # result, error = apply_kNN_movie(int(threshold), str(choice_of_algo), int(choice_of_k), int(choice_of_n))
            # get_top_n(result)
            st.write("Placeholder")

    if page_selection == "Neural network matrix factorization":
        choice_dataset = st.selectbox('Please select the dataset:', ('Amazon reviews', 'Movielens movie reviews'))

        if choice_dataset == 'Amazon reviews':
            path_to_dataset = "../data/ratings_amazon.csv"
            cols = [0, 1, 2]
        elif choice_dataset == 'Movielens movie reviews':
            path_to_dataset = "../data/ml-latest-small/ratings.csv"
            cols = ['userId', 'movieId', 'rating']

        # load ds from utils.py
        df = load_dataset(path=path_to_dataset, cols=cols)
        st.write("Shape of Data", df.shape)
        st.write(df.head(20))

        # parameters fro NNMF
        n_epochs = st.sidebar.slider('Epochs', 1, 20)
        choice_metrics = st.sidebar.selectbox('Which metrics?', ('mae', 'mse'))
        choice_optimizer = st.sidebar.selectbox('Optimizer', ('adam', 'sgd'))
        if st.sidebar.button('Run NNMF Algo'):
            # history, results, model = run_neural_network(1)
            # st.write('model: ', model.metrics_names)
            # st.bar_chart(history.history['loss'])
            # st.bar_chart(history.history['mae'])
            st.write("TEST")


if __name__ == '__main__':
    main()
