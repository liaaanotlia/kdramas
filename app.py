import streamlit as st
import pandas as pd
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("kdrama_cleaned.csv")
    df.columns = df.columns.str.strip()  # Hapus spasi di sekitar nama kolom
    df['Genre'] = df['Genre'].apply(lambda x: x.split())  # Pisahkan genre berdasarkan spasi
    return df

# Fungsi untuk menghitung dot product secara manual
def dot_product(vec_a, vec_b):
    return np.dot(vec_a, vec_b)

# Fungsi untuk menghitung norma (magnitude) secara manual
def vector_norm(vec):
    return np.linalg.norm(vec)

# Fungsi untuk menghitung cosine similarity secara manual
def cosine_similarity_manual(vec_a, vec_b):
    dot = dot_product(vec_a, vec_b)
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

# Judul Aplikasi
st.title("🎥 K-Drama Recommendation")
st.markdown("Discover your next favorite Korean drama with our recommendations! 💖")

# Load dataset
df = load_data()

# Dropdown untuk memilih drama
selected_drama = st.selectbox(
    "🎬 Select a Korean Drama:",
    options=df['Name'].values
)

# Detail drama yang dipilih
drama_detail = df[df['Name'] == selected_drama].iloc[0]

st.write(f"**🎥 Name:** {drama_detail['Name']}")
st.write(f"**📅 Year of Release:** {drama_detail['Year of release'] if 'Year of release' in drama_detail else 'Data not available'}")
st.write(f"**🎞️ Number of Episodes:** {drama_detail['Number of Episodes'] if 'Number of Episodes' in drama_detail else 'Data not available'}")
st.write(f"**⭐ Rating:** {drama_detail['Rating'] if 'Rating' in drama_detail else 'Data not available'}")
st.write(f"**📚 Genre:** {', '.join(drama_detail['Genre'])}")
st.write(f"**👥 Cast:** {drama_detail['Cast'] if 'Cast' in drama_detail else 'Data not available'}")

# Menghitung vektor genre untuk setiap drama
def compute_genre_vector(drama_genre, all_genres):
    return [1 if genre in drama_genre else 0 for genre in all_genres]

# Semua genre unik
all_genres = list({genre for genres in df['Genre'] for genre in genres})

# Vektor genre drama yang dipilih
selected_genre_vector = compute_genre_vector(drama_detail['Genre'], all_genres)

# Menghitung cosine similarity untuk genre
df['genre_similarity'] = df.apply(
    lambda x: cosine_similarity_manual(
        compute_genre_vector(x['Genre'], all_genres),
        selected_genre_vector
    ),
    axis=1
)
# Menghitung vektor cast untuk setiap drama
def compute_cast_vector(drama_cast, all_casts):
    # Pisahkan cast berdasarkan koma dan spasi, dan pastikan elemen yang unik
    cast_list = [cast.strip() for cast in drama_cast.split(",")]
    return [1 if cast in cast_list else 0 for cast in all_casts]

# Semua cast unik
all_casts = list({cast for casts in df['Cast'] for cast in casts})

# Vektor cast drama yang dipilih
selected_cast_vector = compute_cast_vector(drama_detail['Cast'], all_casts)

# Menghitung cosine similarity untuk cast
df['cast_similarity'] = df.apply(
    lambda x: cosine_similarity_manual(
        compute_cast_vector(x['Cast'], all_casts),
        selected_cast_vector
    ),
    axis=1
)

# Menghitung cosine similarity untuk genre & cast
def compute_combined_vector(genre, cast, all_genres, all_casts):
    genre_vector = compute_genre_vector(genre, all_genres)
    cast_vector = [1 if actor in cast.split(", ") else 0 for actor in all_casts]
    return genre_vector + cast_vector

# Daftar semua aktor yang unik (untuk cast vector)
all_casts = list(set([actor for cast in df['Cast'] for actor in cast.split(", ")]))

# Vektor gabungan drama yang dipilih
selected_combined_vector = compute_combined_vector(drama_detail['Genre'], drama_detail['Cast'], all_genres, all_casts)

# Menghitung cosine similarity untuk total (genre + cast)
df['total_similarity'] = df.apply(
    lambda x: cosine_similarity_manual(
        compute_combined_vector(x['Genre'], x['Cast'], all_genres, all_casts),
        selected_combined_vector
    ),
    axis=1
)

# Hapus drama yang dipilih dari daftar rekomendasi
df = df[df['Name'] != selected_drama]

# Rekomendasi berdasarkan genre (hanya yang memiliki genre_similarity > 0)
recommended_by_genre = df[df['genre_similarity'] > 0].sort_values(by='genre_similarity', ascending=False).head(6)

# Rekomendasi berdasarkan cast (hanya yang memiliki cast_similarity > 0)
recommended_by_cast = df[df['cast_similarity'] > 0].sort_values(by='cast_similarity', ascending=False).head(6)

# Rekomendasi berdasarkan genre + cast (hanya yang memiliki total_similarity > 0)
recommended_by_genre_and_cast = df[df['total_similarity'] > 0].sort_values(by='total_similarity', ascending=False).head(6)

# Fungsi untuk menampilkan rekomendasi dalam layout grid (3 kolom) tanpa memotong judul
def display_recommendations(title, recommendations, similarity_col):
    st.subheader(title)
    cols = st.columns(3)  # Layout 3 kolom
    for index, (_, drama) in enumerate(recommendations.iterrows()):
        col = cols[index % 3]  # Distribusi ke kolom berdasarkan index
        with col:
            with st.container():  # Kontainer untuk memastikan elemen sejajar
                st.markdown(f"<h4 style='font-size: 20px; font-weight: bold;'>{drama['Name']}</h4>", unsafe_allow_html=True)
                st.write(f"**⭐ Rating:** {drama['Rating'] if 'Rating' in drama else 'N/A'}")
                st.write(f"**🎞️ Episodes:** {drama['Number of Episodes'] if 'Number of Episodes' in drama else 'N/A'}")
                st.write(f"**📚 Genre:** {', '.join(drama['Genre'])}")
                st.write(f"**👥 Cast:** {drama['Cast'] if 'Cast' in drama else 'N/A'}")  # Menampilkan cast
                st.write(f"**✨ Similarity:** {drama[similarity_col]:.2f}")
                st.markdown("  ")  # Pemisah antar kolom

# Menampilkan rekomendasi
display_recommendations("✨ Recommended K-Dramas Based on Genre and Cast", recommended_by_genre_and_cast, "total_similarity")
display_recommendations("📚 Recommended K-Dramas Based on Genre", recommended_by_genre, "genre_similarity")
display_recommendations("👥 Recommended K-Dramas Based on Cast", recommended_by_cast, "cast_similarity")
