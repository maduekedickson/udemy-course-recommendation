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

# Function to calculate cosine similarity matrix
def get_cosine_similarity_matrix(df):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(df['Clean_title'])
    return cosine_similarity(cv_mat)

# Function to recommend courses based on title
def recommend_courses(df, title, cosine_mat, num_rec=6):
    course_index = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    index = course_index.get(title, None)

    if index is None:
        return None
    
    scores = list(enumerate(cosine_mat[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    selected_course_indices = [i[0] for i in sorted_scores[1:num_rec+1]]
    recommended_df = df.iloc[selected_course_indices]
    return recommended_df

# Load and clean data
df = pd.read_csv('UdemyCleanedTitle.csv')
df = clean_course_titles(df)
cosine_sim_mat = get_cosine_similarity_matrix(df)

# Streamlit App Layout
st.title("Course Recommendation System")

# Display the logo or image
st.image("https://via.placeholder.com/800x300.png?text=Course+Recommendations", use_column_width=True)

# User input for course title
course_title_input = st.text_input("Enter a course keyword or title", "")

# Recommend button
if st.button("Recommend"):
    if course_title_input:
        # Get recommendations
        recommended_df = recommend_courses(df, course_title_input, cosine_sim_mat)
        
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

