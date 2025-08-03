
import streamlit as st
import pandas as pd
from transformers import pipeline
from fer import FER
import cv2
import tempfile

# Load emotion classifier (text)
text_emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=False)

# Load songs database
songs = pd.read_csv("songs.csv")

# Emotion-to-genre map
emotion_to_genre = {
    "joy": ["pop", "indie", "upbeat"],
    "sadness": ["soft rock", "piano", "lofi"],
    "anger": ["metal", "rap"],
    "fear": ["ambient", "instrumental"],
    "neutral": ["lofi", "chill"],
    "surprise": ["edm", "upbeat"],
    "disgust": ["jazz", "classical"],
    "love": ["romantic", "acoustic"],
    "optimism": ["pop", "edm"],
    "grief": ["piano", "soft rock"],
    "admiration": ["indie", "lofi"],
    "amusement": ["pop", "edm"],
}

st.title("🎧 Emotion-Based Music Recommender")

choice = st.radio("Choose Input Type", ["Text", "Image"])

detected_emotion = None

# TEXT INPUT
if choice == "Text":
    user_input = st.text_input("How are you feeling?")
    if st.button("Detect Emotion (Text)"):
        if user_input.strip():
            detected_emotion = text_emotion_classifier(user_input)[0]['label']
            st.success(f"Detected Emotion: **{detected_emotion}**")

# IMAGE INPUT
elif choice == "Image":
    uploaded_file = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, width=250, caption="Uploaded Image")
        if st.button("Detect Emotion (Image)"):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            img = cv2.imread(temp_file.name)

            detector = FER(mtcnn=True)
            detected_emotion, score = detector.top_emotion(img)
            if detected_emotion:
                st.success(f"Detected Emotion: **{detected_emotion}**")
            else:
                st.error("Could not detect emotion from the image.")

# Show recommendations
if detected_emotion:
    genres = emotion_to_genre.get(detected_emotion.lower(), ["pop", "lofi"])  # fallback
    matched_songs = songs[songs['genre'].isin(genres)]

    if not matched_songs.empty:
        st.markdown("### 🎵 Recommended Songs:")
        for _, row in matched_songs.sample(n=min(5, len(matched_songs))).iterrows():
            st.markdown(f"- [{row['title']} - {row['artist']}]({row['url']})")
    else:
        st.warning("No songs found for this emotion.")
