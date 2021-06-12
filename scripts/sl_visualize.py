import streamlit as st
import yfinance as yf

from nnmf_movie_rec import model
st.write("""
# My first app!
Hello, world!
""")

choice_algo = st.selectbox('Please choose the algorithm:', ('NNMF', 'KNN'))
st.write("your Choice: ", choice_algo)
choice_dataset = st.selectbox('Please select the dataset:', ('Amazon reviews', 'Movielens movie reviews'))
st.write('your choice: ', choice_dataset)
number = st.slider("Popularity Threshold", 1, 5,)
st.write("Thresold: ", number)

threshold = st.number_input('Threshold: ', value=.0)
st.write('Threshold: ', threshold)

ticker_symbol = 'BB'

ticker_data = yf.Ticker(ticker_symbol)
ticker_df = ticker_data.history(period='1y', start='2020-6-2', end='2021-6-3')
st.line_chart(ticker_df.Open)
st.line_chart(ticker_df.Close)
st.line_chart(ticker_df.Volume)

@st.cache
def streamlit():
    st.write("""
    # My first app!
    Hello, world!
    """)
