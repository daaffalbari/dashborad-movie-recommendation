import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
import sklearn.feature_extraction.text as TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="TMDB", page_icon="🍿", layout="wide")    
st.markdown(f"""
            <style>
            .stApp {{background-image: url(""); 
                     background-attachment: fixed;
                     base: light;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# Load models and MovieDB
df = joblib.load('../models/movie_db.df')
tfidf_matrix = joblib.load('../models/tfidf_matrix.tf')
tfidf = joblib.load('../models/vectorizer.tf')
cos_mat = joblib.load('../models/cos_mat.mt')


# define functions
def get_keywords_recommendations(keywords):
    
    keywords = keywords.split()
    keywords = " ".join(keywords)    
    # transform the string to vector representation
    key_tfidf = tfidf.transform([keywords]) 
    # compute cosine similarity    
    result = cosine_similarity(key_tfidf, tfidf_matrix)
    # sort top n similar movies   
    similar_key_movies = sorted(list(enumerate(result[0])), reverse=True, key=lambda x: x[1])
    # extract names from dataframe and return movie names
    recomm = []
    for i in similar_key_movies[1:6]:
        recomm.append(df.iloc[i[0]].title)
    return recomm


def get_recommendations(movie):
    
    # get index from dataframe
    index = df[df['title']== movie].index[0]    
    # sort top n similar movies     
    similar_movies = sorted(list(enumerate(cos_mat[index])), reverse=True, key=lambda x: x[1]) 
    # extract names from dataframe and return movie names
    recomm = []
    for i in similar_movies[1:6]:
        recomm.append(df.iloc[i[0]].title)
    return recomm


def fetch_poster(movies):
    
    API_KEY = os.getenv("API_KEY")

    ids = []
    posters = []
    for i in movies:
        ids.append(df[df.title==i]['id'].values[0])

    for i in ids:    
      url =  f"https://api.themoviedb.org/3/movie/{i}?api_key={API_KEY}&language=en-US"
      data = requests.get(url)
      data = data.json()
      poster_path = data['poster_path']
      full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
      posters.append(full_path)
    return posters


# App Layout
st.image("../img/applogo1.png")
st.title("Movie Finder 🍿 🤖")
posters = 0
movies = 0


with st.sidebar:
    st.image("../img/app1.png", use_column_width=True)
    st.header("Get Recommendations by 👇")
    search_type = st.radio("", ('Movie Title', 'Keywords'))
    st.header("Source Code 📦")
    st.markdown("[GitHub Repository](https://github.com/daaffalbari)")    
    st.header("Blog 📝")
    st.markdown("[Medium Article](https://medium.com/@daffabercerita)")

# call functions based on selectbox
if search_type == 'Movie Title': 
    st.subheader("Select Movie 🎬")   
    movie_name = st.selectbox('', df.title)
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies = get_recommendations(movie_name)
            posters = fetch_poster(movies)        
else:
    st.subheader('Enter Cast / Crew / Tags / Genre  🌟')
    keyword = st.text_input('', 'Christopher Nolan')
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies = get_keywords_recommendations(keyword)
            posters = fetch_poster(movies)
              
# display posters       
if posters:
    col1, col2, col3, col4, col5 = st.columns(5, gap='medium')
    with col1:
        st.text(movies[0])
        st.image(posters[0])
    with col2:
        st.text(movies[1])
        st.image(posters[1])

    with col3:
        st.text(movies[2])
        st.image(posters[2])
    with col4:
        st.text(movies[3])
        st.image(posters[3])
    with col5:
        st.text(movies[4])
        st.image(posters[4])
