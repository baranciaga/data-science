import streamlit as st

st.write("""
# My first app!
Hello, world!
""")
number = st.slider("Pick a number", 0, 69,)
date = st.date_input('Pick a Date!')

@st.cache
def streamlit():
    st.write("""
    # My first app!
    Hello, world!
    """)
