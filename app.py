import streamlit as st
import pandas as pd
import random
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎌",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""

<style>

/* MAIN BACKGROUND */
.stApp {
            

    background:
        linear-gradient(
            rgba(0,0,0,0.72),
            rgba(0,0,0,0.72)
        ),
        url("https://i.pinimg.com/736x/0b/0b/ab/0b0babdc7363ca598779f4456348e209.jpg");

    background-size: cover;

    background-position: center;

    background-repeat: no-repeat;

    background-attachment: fixed;
            

    color: white;
}

/* TITLE */
.main-title {

    text-align: center;

    font-size: 52px;

    font-weight: bold;

    color: #facc15;

    text-shadow: 3px 3px 15px black;

    margin-top: 10px;
}

/* SUBTITLE */
.subtitle {

    text-align: center;

    font-size: 18px;

    color: #f3f4f6;

    margin-bottom: 30px;
}

/* RECOMMENDATION CARD */
.anime-box {

    background: rgba(255,255,255,0.10);

    backdrop-filter: blur(12px);

    -webkit-backdrop-filter: blur(12px);

    padding: 20px;

    border-radius: 22px;

    margin-bottom: 20px;

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow: 0 8px 25px rgba(0,0,0,0.4);

    color: white;

    transition: 0.3s;
}

/* CARD HOVER */
.anime-box:hover {

    transform: scale(1.02);
}

/* BUTTON */
.stButton>button {

    width: 100%;

    background: linear-gradient(
        to right,
        #ec4899,
        #8b5cf6
    );

    color: white;

    border: none;

    border-radius: 14px;

    padding: 12px;

    font-size: 18px;

    font-weight: bold;

    transition: 0.3s;

    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

/* BUTTON HOVER */
.stButton>button:hover {

    transform: scale(1.03);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {

    background: rgba(0,0,0,0.45);

    backdrop-filter: blur(10px);
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] {

    background: rgba(255,255,255,0.08);

    border-radius: 12px;

    color: white;
}

/* FOOTER */
.footer {

    text-align: center;

    margin-top: 40px;

    color: #d1d5db;

    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown(
    '<div class="main-title">🎌 Anime Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover your next favorite anime ✨</div>',
    unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This system uses:

    ✅ Content-Based Filtering  
    ✅ Cosine Similarity  
    ✅ Streamlit UI  
    ✅ Kaggle Dataset
    """
)

# TRENDING SECTION
st.sidebar.title("🔥 Trending Anime")

st.sidebar.write("""
⭐ Attack on Titan  
⭐ Demon Slayer  
⭐ Jujutsu Kaisen  
⭐ One Piece  
⭐ Naruto
""")

# GENRE SECTION
st.sidebar.title("🎭 Popular Genres")

st.sidebar.write("""
⚔ Action  
💕 Romance  
😂 Comedy  
🌌 Fantasy  
👻 Horror
""")

# SLIDER
num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    1,
    10,
    5
)

# RANDOM QUOTES
quotes = [
    "Power comes in response to a need.",
    "Fear is not evil. It tells you your weakness.",
    "A lesson without pain is meaningless.",
    "People’s lives don’t end when they die.",
    "Whatever you lose, you'll find it again."
]

st.markdown(
    f"""
    <div style="
        background: rgba(255,255,255,0.12);
        padding: 18px;
        border-radius: 15px;
        font-size: 22px;
        font-weight: bold;
        color: white;
        text-align: center;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    ">
    ✨ {random.choice(quotes)}
    </div>
    """,
    unsafe_allow_html=True
)

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv(
        "small_anime.csv",
        engine='python'
    )

data = load_data()

data = data.dropna()

# FEATURES
data["features"] = data["genre"] + " " + data["type"]

# VECTORIZE
cv = CountVectorizer(stop_words='english')

matrix = cv.fit_transform(data["features"])

# SIMILARITY
similarity = cosine_similarity(matrix)

# ANIME LIST
anime_list = data["name"].values

# POSTERS
poster_dict = {

    "Naruto":
    "https://cdn.myanimelist.net/images/anime/13/17405.jpg",

    "One Piece":
    "https://cdn.myanimelist.net/images/anime/6/73245.jpg",

    "Death Note":
    "https://cdn.myanimelist.net/images/anime/9/9453.jpg",

    "Attack on Titan":
    "https://cdn.myanimelist.net/images/anime/10/47347.jpg",

    "Demon Slayer":
    "https://cdn.myanimelist.net/images/anime/1286/99889.jpg",

    "Jujutsu Kaisen":
    "https://cdn.myanimelist.net/images/anime/1171/109222.jpg",

    "Tokyo Ghoul":
    "https://cdn.myanimelist.net/images/anime/5/64449.jpg",

    "Bleach":
    "https://cdn.myanimelist.net/images/anime/3/40451.jpg",

    "Dragon Ball Z":
    "https://cdn.myanimelist.net/images/anime/6/20936.jpg",

    "Hunter x Hunter":
    "https://cdn.myanimelist.net/images/anime/1337/99013.jpg",

    "Hunter x Hunter (2011)":
    "https://cdn.myanimelist.net/images/anime/1337/99013.jpg",

    "Your Name":
    "https://cdn.myanimelist.net/images/anime/5/87048.jpg",

    "Kimi no Na wa.":
    "https://cdn.myanimelist.net/images/anime/5/87048.jpg",

    "Spirited Away":
    "https://cdn.myanimelist.net/images/anime/6/79597.jpg",

    "Haikyuu":
    "https://cdn.myanimelist.net/images/anime/7/76014.jpg",

    "Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou":
    "https://cdn.myanimelist.net/images/anime/7/81919.jpg",

    "Fairy Tail":
    "https://cdn.myanimelist.net/images/anime/5/18179.jpg",

    "Fullmetal Alchemist: Brotherhood":
    "https://cdn.myanimelist.net/images/anime/1223/96541.jpg",

    "Steins;Gate":
    "https://cdn.myanimelist.net/images/anime/5/73199.jpg",

    "Gintama°":
    "https://cdn.myanimelist.net/images/anime/3/72078.jpg",

    "Gintama&#039;":
    "https://cdn.myanimelist.net/images/anime/4/50361.jpg"
}
# SELECT BOX
selected_anime = st.selectbox(
    "🎥 Select Anime",
    anime_list,
    index=None,
    placeholder="Choose Anime"
)

# SHOW POSTER
if selected_anime:

    anime_key = selected_anime.strip()

    if anime_key in poster_dict:

        st.image(
            poster_dict[anime_key],
            width=280
        )

# RECOMMEND FUNCTION
def recommend(anime):

    index = data[data["name"] == anime].index[0]

    distances = list(enumerate(similarity[index]))

    sorted_distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in sorted_distances[1:num_recommendations+1]:

        anime_data = data.iloc[i[0]]

        similarity_score = round(i[1] * 100, 2)

        recommendations.append(
            (anime_data, similarity_score)
        )

    return recommendations

# BUTTON
if st.button("✨ Recommend Anime"):

    if selected_anime is None:

        st.warning("Please select an anime first!")

    else:

        with st.spinner("Finding best anime for you..."):

            time.sleep(2)

            recommendations = recommend(selected_anime)

        st.subheader("🔥 Recommended Anime")

        for anime, score in recommendations:

            anime_name = anime['name']

            st.markdown(f"""
            <div class="anime-box">
                <h3>🎬 {anime_name}</h3>
                <p><b>Genre:</b> {anime['genre']}</p>
                <p><b>Type:</b> {anime['type']}</p>
                <p><b>Rating:</b> ⭐ {anime['rating']}</p>
                <p><b>Match:</b> 🔥 {score}%</p>
            </div>
            """, unsafe_allow_html=True)

            if anime_name in poster_dict:

                st.image(
                    poster_dict[anime_name],
                    width=200
                )

# FOOTER
st.markdown(
    '<div class="footer">Made with ❤️ using Streamlit</div>',
    unsafe_allow_html=True
)
