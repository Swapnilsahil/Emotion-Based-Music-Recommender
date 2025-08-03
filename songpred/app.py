import streamlit as st
import pandas as pd
from transformers import pipeline
from fer import FER
import cv2
import tempfile
from PIL import Image

# Load sentiment analysis model
st.set_page_config(page_title="Emotion-Based Music Recommender", layout="centered")
st.title("🎧 Emotion-Based Music Recommender")
st.markdown("Upload an image or type how you feel, and we'll suggest songs that match your mood!")

# Load data
songs = pd.read_csv("songs.csv")

# Emotion to genre mapping
emotion_to_genre = {
    "happy": ["pop", "dance"],
    "sad": ["acoustic", "piano"],
    "angry": ["rock", "metal"],
    "fear": ["ambient", "instrumental"],
    "surprise": ["electronic", "alternative"],
    "disgust": ["grunge", "dark"],
    "neutral": ["indie", "folk"]
}

# ---------------- Text-Based Emotion ---------------- #
st.header("📝 Text-Based Emotion")
user_input = st.text_input("Describe your mood (e.g., 'I feel great today!')")

if user_input:
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(user_input)[0]
    st.write(f"Detected Emotion: **{result['label'].lower()}**")

    emotion = result['label'].lower()
    genre_list = emotion_to_genre.get(emotion, ["pop"])  # fallback to pop

    recommendations = songs[songs['genre'].isin(genre_list)].sample(5, replace=True)
    st.subheader("🎵 Recommended Songs:")
    st.table(recommendations[["title", "artist", "genre"]])


# ---------------- Image-Based Emotion ---------------- #
st.header("📷 Image-Based Emotion")
uploaded_image = st.file_uploader("Upload a selfie or facial photo", type=["jpg", "png", "jpeg"])

if uploaded_image:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_image.read())
        tmp_path = tmp_file.name

    image = cv2.imread(tmp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = FER(mtcnn=True)
    emotions = detector.detect_emotions(image_rgb)

    if emotions:
        top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        st.image(image_rgb, caption=f"Detected Emotion: {top_emotion}", use_column_width=True)

        genre_list = emotion_to_genre.get(top_emotion, ["pop"])
        recommendations = songs[songs['genre'].isin(genre_list)].sample(5, replace=True)
        st.subheader("🎵 Recommended Songs:")
        st.table(recommendations[["title", "artist", "genre"]])
    else:
        st.warning("Couldn't detect any face. Try uploading a clearer image.")

