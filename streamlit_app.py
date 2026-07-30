import streamlit as st
import pandas as pd
from streamlit_mic_recorder import speech_to_text

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="EmoTune AI",
    page_icon="🎵",
    layout="wide"
)

# ==============================
# MACHINE LEARNING
# ==============================

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def train_model():

    df = pd.read_csv("emotions.csv")

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])

    model.fit(df["text"], df["emotion"])

    return model

model = train_model()

# ==============================
# CSS DESIGN
# ==============================

st.markdown("""
<style>

/* =========================
   FULL PAGE BACKGROUND
========================= */

.stApp {
    background: linear-gradient(135deg,#20c4f4,#001b7a);
}

/* =========================
   REMOVE EXTRA SPACE
========================= */

.block-container {
    padding-top: 1rem !important;
    padding-left: 5rem !important;
    padding-right: 5rem !important;
}

/* =========================
   HIDE STREAMLIT HEADER
========================= */

header {
    visibility: hidden;
}

/* =========================
   TITLE
========================= */

h1 {
    text-align: center !important;
    font-size: 60px !important;
    color: white !important;
    font-weight: 800 !important;
    margin-bottom: 10px !important;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.4);
}

/* =========================
   SUBTITLE
========================= */

.subtitle {
    text-align: center;
    font-size: 22px;
    color: white !important;
    font-weight: 600;
    margin-bottom: 40px;
}

/* =========================
   LABEL
========================= */

label {
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* =========================
   INPUT BOX
========================= */

.stTextInput > div > div > input {

    background: rgba(255,255,255,0.95) !important;

    color: black !important;

    border-radius: 14px !important;

    padding: 14px !important;

    font-size: 18px !important;

    border: none !important;

    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

/* =========================
   PLACEHOLDER
========================= */

.stTextInput input::placeholder {
    color: #374151 !important;
    opacity: 1 !important;
}

/* =========================
   BUTTON
========================= */

.stButton > button {

    background: linear-gradient(90deg,#ff416c,#ff4b2b) !important;

    color: white !important;

    border: none !important;

    border-radius: 14px !important;

    padding: 12px 28px !important;

    font-size: 18px !important;

    font-weight: bold !important;

    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);

    transition: 0.3s;
}

/* BUTTON HOVER */

.stButton > button:hover {
    transform: scale(1.05);
}

/* =========================
   MOOD CARD
========================= */

.card {

    background: rgba(255,255,255,0.12);

    backdrop-filter: blur(12px);

    padding: 25px;

    border-radius: 20px;

    margin-top: 25px;

    color: white;

    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

/* =========================
   SONG BOX
========================= */

.song {

    background: rgba(255,255,255,0.18);

    padding: 12px;

    border-radius: 12px;

    margin-top: 10px;

    color: white;

    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================

st.title("🎵 EmoTune AI")

st.markdown(
    "<div class='subtitle'>Voice + ML Mood Detection + YouTube Songs</div>",
    unsafe_allow_html=True
)

# ==============================
# TEXT INPUT
# ==============================

text = st.text_input(
    "💭 Enter your mood",
    placeholder="Type your feelings here..."
)

# ==============================
# VOICE INPUT
# ==============================

voice_text = speech_to_text(
    language='en',
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording",
    use_container_width=True,
    just_once=True,
    key='STT'
)

if voice_text:

    text = voice_text

    st.success(f"🎙 Detected Voice: {voice_text}")

# ==============================
# PREDICTION
# ==============================

if text:

    prediction = model.predict([text])[0]

    # ==========================
    # HAPPY
    # ==========================

    if prediction == "happy":

        st.balloons()

        mood_emoji = "😊"

        mood_message = "Happiness detected! Keep smiling ✨"

        songs = [
            "🎵 Happy Playlist",
            "🎵 Motivation Songs",
            "🎵 Party Hits"
        ]

        yt_link = "https://www.youtube.com/results?search_query=happy+songs"

        bg_color = "#22c55e"

    # ==========================
    # SAD
    # ==========================

    elif prediction == "sad":

        st.snow()

        mood_emoji = "😢"

        mood_message = "Sad mood detected 💙 Everything will be okay."

        songs = [
            "🎵 Sad Songs",
            "🎵 Healing Music",
            "🎵 Calm Playlist"
        ]

        yt_link = "https://www.youtube.com/results?search_query=Emotional+%26+Melancholic+songs"

        bg_color = "#3b82f6"

    # ==========================
    # ANGRY
    # ==========================

    elif prediction == "angry":

        mood_emoji = "😡"

        mood_message = "Angry mood detected 🔥 Relax yourself."

        songs = [
            "🎵 Calm Music",
            "🎵 Relax Beats",
            "🎵 Meditation Songs"
        ]

        yt_link = "https://www.youtube.com/results?search_query=calm+music"

        bg_color = "#ef4444"

        st.warning("😡 Calm Down & Relax")

    # ==========================
    # FEAR
    # ==========================

    elif prediction == "fear":

        mood_emoji = "😨"

        mood_message = "Fear detected 😰 Stay strong."

        songs = [
            "🎵 Relaxing Music",
            "🎵 Peaceful Songs",
            "🎵 Stress Relief Music"
        ]

        yt_link = "https://www.youtube.com/results?search_query=strong+songs"

        bg_color = "#8b5cf6"

        st.info("🌿 Relax Your Mind")

    # ==========================
    # SURPRISE
    # ==========================

    elif prediction == "surprise":

        st.balloons()

        mood_emoji = "😲"

        mood_message = "Surprise mood detected 🎉"

        songs = [
            "🎵 Trending Songs",
            "🎵 Celebration Songs",
            "🎵 Viral Hits"
        ]

        yt_link = "https://www.youtube.com/results?search_query=celebration+songs"

        bg_color = "#f59e0b"

    # ==========================
    # NEUTRAL
    # ==========================

    else:

        mood_emoji = "🙂"

        mood_message = "Neutral mood detected 🌙"

        songs = [
            "🎵 Chill Music",
            "🎵 Soft Songs",
            "🎵 Lofi Playlist"
        ]

        yt_link = "https://www.youtube.com/results?search_query=neutral+song"

        bg_color = "#64748b"

    # ==============================
    # RESULT CARD
    # ==============================

    st.markdown(f"""
    <div style="
        background:{bg_color};
        padding:25px;
        border-radius:20px;
        margin-top:20px;
        color:white;
        text-align:center;
        box-shadow:0px 4px 20px rgba(0,0,0,0.3);
    ">
        <h1>{mood_emoji} {prediction.upper()}</h1>
        <h3>{mood_message}</h3>
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    # SONGS
    # ==============================

    st.markdown("## 🎶 Recommended Songs")

    for song in songs:

        st.markdown(f"""
        <div style="
            background:rgba(255,255,255,0.15);
            padding:12px;
            border-radius:12px;
            margin-top:10px;
            color:white;
            font-size:18px;
        ">
            {song}
        </div>
        """, unsafe_allow_html=True)

    # ==============================
    # YOUTUBE BUTTON
    # ==============================

    st.link_button(
        "▶ Play Songs on YouTube",
        yt_link
    )