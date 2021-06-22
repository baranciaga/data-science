import streamlit as st
from knn_movie_rec import apply_kNN_movie, get_top_n
import pandas as pd
from utils import load_dataset, get_value_counts
# from nnmf_movie_rec import run_neural_network, recommend
from tensorflow import keras
import numpy as np



def main():
    """
    main method of the streamlit UI
    Returns: None
    """
    # title
    st.write("""
    # Barans and Dungs Recommender
    """)
    # sidebar options
    choice_dataset = st.sidebar.selectbox('Please select the dataset:', ('Amazon reviews', 'Movielens movie reviews'))
    threshold = st.sidebar.number_input('Threshold: ', value=100)
    choice_of_algo = st.sidebar.selectbox("Which Similarity metric?", ("pearson", "cosine"))


    if choice_dataset == 'Amazon reviews':

        path_to_dataset = "../data/ratings_amazon.csv"
        cols = ['userId', 'itemId', 'rating']

        # load dataset
        df = load_dataset(path=path_to_dataset, cols=cols)

        st.write("Head of dataframe: ")
        st.write(df.head(20))

        # Variance of the dataset as bar chart
        st.write('Number of ratings:')
        s = get_value_counts(df)
        df = pd.DataFrame(s).T
        st.bar_chart(df)
        df_amazon = pd.read_csv('../data/ratings_amazon.csv', header=None, usecols=[0, 1, 2],
                                names=['userId', 'productId', 'rating'], nrows=50000)
        df_amazon = df_amazon.pivot(index='userId', columns='productId', values='rating')

        st.write('pivoted data: ', df_amazon.head(20))

    elif choice_dataset == 'Movielens movie reviews':
        # load dataset
        path_to_dataset = "../data/ml-latest-small/ratings.csv"
        cols = ['userId', 'movieId', 'rating']
        df = load_dataset(path=path_to_dataset, cols=cols)
        st.write("Head of dataframe: ")
        st.write(df.head(20))

        # barchart of unique ratings -> variance
        st.write('Number of ratings:')
        s = get_value_counts(df)
        df = pd.DataFrame(s).T
        st.bar_chart(df)
        print("Ratings Df: ", df)

        st.write("Shape of Data", df.shape)
        st.write(df.head(20))

        dataset1 = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=['userId', 'movieId', 'rating'])
        dataset1 = dataset1.pivot(index='userId', columns='movieId', values='rating')

        st.write('pivoted data: ', dataset1.head(20))

    # output of data




    # button for knn algo
    if st.sidebar.button('Run algorithms'):
        top_n, error = apply_kNN_movie(int(threshold), str(choice_of_algo), False)
        # get_top_n(result)
        st.write("Placeholder")

        # loading the history of the nnmf model with mae and loss information
        history = np.load('../data/model_data/ml/history.npy', allow_pickle='TRUE').item()

        st.write("MAE over 12 epochs")
        # st.write('model: ', model.metrics_names)
        df_bar = pd.DataFrame([history['val_mae'][11], error]) # , columns=['NNMF', 'KNN']

        st.bar_chart(df_bar, width=600)

        st.write(top_n[1])
        new = pd.DataFrame.from_dict(top_n, orient='index')
        st.write(new.head())
        print(new.head())

if __name__ == '__main__':
    main()
