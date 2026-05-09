import streamlit as st
import pandas as pd
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

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #1e293b, #312e81);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #facc15;
    margin-bottom: 10px;
    text-shadow: 2px 2px 10px black;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #d1d5db;
    margin-bottom: 30px;
}

.anime-box {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
}

.recommend-title {
    color: #f472b6;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.stButton>button {
    background: linear-gradient(to right, #ec4899, #8b5cf6);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(to right, #f43f5e, #7c3aed);
}

div[data-baseweb="select"] {
    background-color: white;
    border-radius: 10px;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #cbd5e1;
    font-size: 14px;
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

# LOAD DATASET
data = pd.read_csv("small_anime.csv", engine='python')

# REMOVE NULL VALUES
data = data.dropna()

# COMBINE FEATURES
data["features"] = data["genre"] + " " + data["type"]

# TEXT TO VECTOR
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(data["features"])

# SIMILARITY
similarity = cosine_similarity(matrix)

# ANIME LIST
anime_list = data["name"].values

# SELECT BOX
selected_anime = st.selectbox(
    "🎥 Select Your Favorite Anime",
    anime_list
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

    for i in sorted_distances[1:6]:

        recommendations.append(
            data.iloc[i[0]]
        )

    return recommendations

# BUTTON
if st.button("✨ Recommend Anime"):

    recommendations = recommend(selected_anime)

    st.markdown(
        '<div class="recommend-title">🔥 Recommended Anime</div>',
        unsafe_allow_html=True
    )

    for anime in recommendations:

        st.markdown(f"""
        <div class="anime-box">
            <h3>🎬 {anime['name']}</h3>
            <p><b>Genre:</b> {anime['genre']}</p>
            <p><b>Type:</b> {anime['type']}</p>
            <p><b>Rating:</b> ⭐ {anime['rating']}</p>
        </div>
        """, unsafe_allow_html=True)

# FOOTER
st.markdown(
    '<div class="footer">Made with ❤️ using Streamlit</div>',
    unsafe_allow_html=True
)