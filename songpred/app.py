import streamlit as st
import pandas as pd
from PIL import Image
import cv2
from fer import FER
from transformers import pipeline

# Load data
@st.cache_data
def load_songs():
    return pd.read_csv("songs.csv")

songs_df = load_songs()

# Emotion-to-genre mapping
emotion_to_genres = {
    "happy": ["pop", "dance", "electronic"],
    "sad": ["acoustic", "blues", "soft rock"],
    "angry": ["rock", "metal"],
    "surprise": ["indie", "funk"],
    "fear": ["ambient", "classical"],
    "disgust": ["grunge", "alternative"],
    "neutral": ["lofi", "chill"]
}

# Title
st.title("🎵 Emotion-Based Music Recommendation System")

# Choose mode
mode = st.radio("Choose Input Mode", ["Text (How you feel)", "Image (Facial Emotion)"])

# 1. Text Input Mode
if mode == "Text (How you feel)":
    user_text = st.text_area("Describe your feeling:")
    if st.button("Detect Emotion from Text"):
        classifier = pipeline("sentiment-analysis")
        result = classifier(user_text)[0]
        emotion = result["label"].lower()
        st.success(f"Detected Emotion: **{emotion}**")

        if emotion in emotion_to_genres:
            st.subheader("🎧 Recommended Songs")
            genre_list = emotion_to_genres[emotion]
            recommendations = songs_df[songs_df['genre'].isin(genre_list)].sample(n=5)
            st.dataframe(recommendations[["title", "artist", "genre"]])
        else:
            st.warning("No songs found for this emotion.")

# 2. Image Upload Mode
else:
    uploaded_file = st.file_uploader("Upload your selfie image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)

        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        detector = FER(mtcnn=True)
        result = detector.top_emotion(img_np)

        if result:
            emotion, score = result
            st.success(f"Detected Emotion: **{emotion}** ({score:.2f})")

            if emotion in emotion_to_genres:
                st.subheader("🎧 Recommended Songs")
                genre_list = emotion_to_genres[emotion]
                recommendations = songs_df[songs_df['genre'].isin(genre_list)].sample(n=5)
                st.dataframe(recommendations[["title", "artist", "genre"]])
            else:
                st.warning("No songs found for this emotion.")
        else:
            st.warning("No emotion detected in image.")
