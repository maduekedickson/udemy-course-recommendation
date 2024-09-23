import streamlit as st
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to clean course titles
def clean_course_titles(df):
    df['Clean_title'] = df['course_title'].apply(nfx.remove_stopwords)
    df['Clean_title'] = df['Clean_title'].apply(nfx.remove_special_characters)
    df['Clean_title'] = df['Clean_title'].str.lower()
    return df

# Function to vectorize titles and calculate cosine similarity matrix
def get_cosine_similarity_matrix(df, query=None):
    count_vect = CountVectorizer()
    
    # Fit the vectorizer on the course titles
    cv_mat = count_vect.fit_transform(df['Clean_title'])
    
    if query is not None:
        # Vectorize the user query the same way
        query_vec = count_vect.transform([query])
        # Calculate cosine similarity between the query and all courses
        cosine_sim = cosine_similarity(query_vec, cv_mat)
        return cosine_sim
    else:
        return cosine_similarity(cv_mat)

# Function to recommend courses based on title
def recommend_courses(df, title, num_rec=6):
    # Clean the input title the same way as the course titles
    clean_title = nfx.remove_stopwords(title)
    clean_title = nfx.remove_special_characters(clean_title)
    clean_title = clean_title.lower()

    # Get the similarity scores for the input title
    cosine_sim = get_cosine_similarity_matrix(df, query=clean_title)

    # Sort the similarity scores
    scores = list(enumerate(cosine_sim[0]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Select the top recommendations (excluding the first result if it's an exact match)
    selected_course_indices = [i[0] for i in sorted_scores[:num_rec]]
    recommended_df = df.iloc[selected_course_indices]
    return recommended_df

# Load and clean data
df = pd.read_csv('UdemyCleanedTitle.csv')
df = clean_course_titles(df)

# Streamlit App Layout
st.title("Udemy Course Recommendation")

# Display the logo or image
st.image("image.jpg", use_column_width=True)

# User input for course title
course_title_input = st.text_input("Enter a course keyword or title", "")

# Recommend button
if st.button("Recommend"):
    if course_title_input:
        # Get recommendations
        recommended_df = recommend_courses(df, course_title_input)
        
        if recommended_df is not None and not recommended_df.empty:
            st.subheader("Recommended Courses")

            # Display recommendations in a grid of 2 columns x 3 rows
            for i in range(0, len(recommended_df), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(recommended_df):
                        course = recommended_df.iloc[i + j]
                        col.write(f"### {course['course_title']}")
                        if course['url']:
                            col.markdown(f"[View Course]({course['url']})")
        else:
            st.warning("No courses found for the given keyword.")
    else:
        st.error("Please enter a course keyword or title.")
