import streamlit as st
import yfinance as yf
st.write("""
# My first app!
Hello, world!
""")
number = st.slider("Pick a number", 0, 69,)
st.write("""
## PHAM ALTER!!! 
""")

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
