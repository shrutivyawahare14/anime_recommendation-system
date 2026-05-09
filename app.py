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
    background-image: url("https://images.unsplash.com/photo-1578632767115-351597cf2477?q=80&w=1974&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.7);
    z-index: -1;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #facc15;
    text-shadow: 2px 2px 10px black;
}

.subtitle {
    text-align: center;
    color: #e5e7eb;
    font-size: 18px;
    margin-bottom: 30px;
}

.anime-box {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    padding: 15px;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #ec4899, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.02);
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #d1d5db;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown(
    '<div class="main-title">🎌 Anime Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find your next favorite anime ✨</div>',
    unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This Anime Recommendation System uses:
    
    ✅ Content-Based Filtering  
    ✅ Cosine Similarity  
    ✅ Streamlit UI  
    ✅ Kaggle Anime Dataset
    """
)

# SLIDER
num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    1,
    10,
    5
)

# LOAD DATASET
data = pd.read_csv("small_anime.csv", engine='python')

data = data.dropna()

# COMBINE FEATURES
data["features"] = data["genre"] + " " + data["type"]

# VECTORIZE
cv = CountVectorizer(stop_words='english')

matrix = cv.fit_transform(data["features"])

# SIMILARITY
similarity = cosine_similarity(matrix)

# ANIME LIST
anime_list = data["name"].values

# POSTER URLs
poster_dict = {
    "Naruto": "https://cdn.myanimelist.net/images/anime/13/17405.jpg",
    "One Piece": "https://cdn.myanimelist.net/images/anime/6/73245.jpg",
    "Death Note": "https://cdn.myanimelist.net/images/anime/9/9453.jpg",
    "Attack on Titan": "https://cdn.myanimelist.net/images/anime/10/47347.jpg",
    "Demon Slayer": "https://cdn.myanimelist.net/images/anime/1286/99889.jpg",
    "Jujutsu Kaisen": "https://cdn.myanimelist.net/images/anime/1171/109222.jpg"
}

# SELECT BOX
selected_anime = st.selectbox(
    "🎥 Select Anime",
    anime_list
)

# SHOW POSTER
if selected_anime in poster_dict:
    st.image(
        poster_dict[selected_anime],
        width=250
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

        recommendations.append(
            data.iloc[i[0]]
        )

    return recommendations

# BUTTON
if st.button("✨ Recommend Anime"):

    recommendations = recommend(selected_anime)

    st.subheader("🔥 Recommended Anime")

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
