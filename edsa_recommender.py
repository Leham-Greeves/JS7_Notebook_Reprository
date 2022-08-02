"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", 'Visual Analysis', 'Contact App Developers']
    st.sidebar.image("resources/imgs/DataSim.png", use_column_width=True)
    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.subheader('Data Sim :star:')
        st.write('We at **Data sim** are a collective of fast moving and future minded software enthusiasts. Even though we may relish in a challange. Our goal is to do so in order you do not have to. Our slogen "Data Simlified" looks at allowing you to enjoy the power of the future!')
        st.subheader('Recommender Systems :gear:')
        st.write("It is estimated that Netflix in the past years made $1 0000 0000 0000 annually on recommender systems annually! Anyone remotely interestedin media would then ask what are recommmender systems. The focus of this app looked at the idea of using recommender systems in order to build recomendations based. These recomenders use a function called cosine similarity. In simple terms hit compares different rows of data in a data set. When it dose it sorts said data into clusters of infomation that one would find. This is particularly important when using this to build a recommender engine.")
        st.subheader('Content Based Recommender :movie_camera:')
        st.write('A Content-Based Recommender works by the data that we take from the user, either explicitly (rating) or implicitly. By the data we create a user profile, which is then used to suggest to the user, as the user provides more input or take more actions on the recommendation, the engine becomes more accurate. [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)')
        st.subheader('Collaborative Based Recommender :film_frames:')
        st.write('In Collaborative Filtering, we tend to find similar users and recommend what similar users like. In this type of recommendation system, we don’t use the features of the item to recommend it, rather we classify the users into the clusters of similar types, and recommend each user according to the preference of its cluster[link](https://www.geeksforgeeks.org/ml-content-based-recommender-system/).')
        st.subheader('Why You Should Consider Being a Partner! :seedling:')
        st.write('Recommender systems are used in a variety of areas, with commonly recognised examples taking the form of playlist generators for video and music services, product recommenders for online stores, or content recommenders for social media platforms and open web content recommenders.[4][5] These systems can operate using a single input, like music, or multiple inputs within and across platforms like news, books and search queries. There are also popular recommender systems for specific topics like restaurants and online dating.')

    if page_selection == "Visual Analysis":
        st.title("Visual Analysis")
        st.header("This is a page that looks at some cool visual understanding of the data!")
        st.subheader('1.Here we look at the distribution of the data and look at how the ratings are spreadout in the different rating units')
        st.image('resources/imgs/output.png')
        st.subheader('2.This visual looks at the idea of the different clusters of movies in the data')
        st.image('resources/imgs/dendogram.png')
        st.subheader('3.Here we learn that in general there are essentially 4 clusters or genres of movies one can collect from the data')
        st.image('resources/imgs/optimal clustering.png')
        st.subheader('4.Recommenders use a form of comparitive analysis between point of data, this illustration looks at the acuracy of the collabritive recommender at different intervals of analysis')
        st.image('resources/imgs/collabrotive based.png')
        st.subheader('5. The focus of this visual looks to understand the number of componants (columns in a dataset) that one can reduce to. Doing so while  still getting a good estimation allowing for efficient datasets. In this illustration we can see that we can essentially remove one of the columns. For This dataset it is the timestamp column ')
        st.image('resources/imgs/Number of componants nessacerry.png')
        st.subheader('6. Curious about the most popular genres are in the data look at our word cloud and let us know what you think!')
        st.image('resources/imgs/wordcloud.png')

    if page_selection == 'Contact App Developers':
        st.header('For any questions or queries, please contact one of our amazing staff:')
        st.subheader('Web designer')
        st.write('Sinethemba: sinethembapurity@gmail.com')

        st.subheader('Machine Learning Engineer')
        st.write('Baby: Babymulaudzi@gmail.com')

        st.subheader('Principal Data Scientist')
        st.write('Leham: leham.greeves@gmail.com')

        st.subheader('Data Engineer')
        st.write('Thato: thatomoepe@gmail.com')
		
        st.subheader('Data Analyst')
        st.write('Phindile: hlelap60@gmail.com')

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.



if __name__ == '__main__':
    main()
