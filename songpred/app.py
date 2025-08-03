import streamlit as st
import pandas as pd
from transformers import pipeline

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Load your songs dataset
songs = pd.read_csv("songpred/songs.csv")

# Map emotions to genres (you can customize this)
emotion_to_genres = {
    "POSITIVE": ["happy", "pop", "dance"],
    "NEGATIVE": ["sad", "blues", "acoustic"]
}

def recommend_songs(emotion_label):
    genres = emotion_to_genres.get(emotion_label.upper(), [])
    recommended = songs[songs['genre'].str.lower().isin(genres)]
    return recommended.sample(min(5, len(recommended))) if not recommended.empty else pd.DataFrame()

# Streamlit UI
st.title("🎵 Emotion-Based Music Recommender")

user_input = st.text_area("Tell us how you feel 👇", placeholder="I'm feeling stressed...")

if st.button("Recommend Songs"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze your mood.")
    else:
        result = sentiment_pipeline(user_input)[0]
        st.write(f"**Detected Emotion**: {result['label']} (Score: {result['score']:.2f})")
        recs = recommend_songs(result['label'])
        if recs.empty:
            st.info("No matching songs found. Try a different mood or add more data.")
        else:
            st.subheader("🎧 Recommended Songs:")
            st.table(recs[['title', 'artist', 'genre']])

