import streamlit as st
import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("kdrama_cleaned.csv")
    df.columns = df.columns.str.strip()  # Hapus spasi di sekitar nama kolom
    df['Genre'] = df['Genre'].apply(lambda x: x.split())  # Pisahkan genre berdasarkan spasi
    return df

# Fungsi untuk menghitung dot product secara manual
def dot_product(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))

# Fungsi untuk menghitung norma (magnitude) secara manual
def vector_norm(vec):
    return sum(x**2 for x in vec) ** 0.5

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

# Menghitung cosine similarity untuk cast
def compute_cast_similarity(cast_a, cast_b):
    cast_list_a = cast_a.split(", ") if isinstance(cast_a, str) else []
    cast_list_b = cast_b.split(", ") if isinstance(cast_b, str) else []
    cast_vector = [1 if actor in cast_list_a else 0 for actor in cast_list_b]
    return cosine_similarity_manual(cast_vector, [1] * len(cast_vector))

df['cast_similarity'] = df.apply(
    lambda x: compute_cast_similarity(drama_detail['Cast'], x['Cast']),
    axis=1
)

# Total similarity berdasarkan genre + cast
df['total_similarity'] = df['genre_similarity'] + df['cast_similarity']

# Hapus drama yang dipilih dari daftar rekomendasi
df = df[df['Name'] != selected_drama]

# Rekomendasi berdasarkan genre
recommended_by_genre = df.sort_values(by='genre_similarity', ascending=False).head(5)

# Rekomendasi berdasarkan cast
recommended_by_cast = df.sort_values(by='cast_similarity', ascending=False).head(5)

# Rekomendasi berdasarkan genre + cast
recommended_by_genre_and_cast = df.sort_values(by='total_similarity', ascending=False).head(5)

# Fungsi untuk menampilkan rekomendasi dalam layout grid (3 kolom) tanpa memotong judul
def display_recommendations(title, recommendations, similarity_col):
    st.subheader(title)
    cols = st.columns(3)  # Layout 3 kolom
    for index, (_, drama) in enumerate(recommendations.iterrows()):
        col = cols[index % 3]  # Distribusi ke kolom berdasarkan index
        with col:
            with st.container():  # Kontainer untuk memastikan elemen sejajar
                # Judul drama dengan highlight tambahan
                st.markdown(f"**🎬 {drama['Name']}**")  # Judul drama tetap utuh
                st.write(f"**⭐ Rating:** {drama['Rating'] if 'Rating' in drama else 'N/A'}")
                st.write(f"**🎞️ Episodes:** {drama['Number of Episodes'] if 'Number of Episodes' in drama else 'N/A'}")
                st.write(f"**📚 Genre:** {', '.join(drama['Genre'])}")
                st.write(f"**✨ Similarity:** {drama[similarity_col]:.2f}")

# Menampilkan rekomendasi
display_recommendations("✨ Recommended K-Dramas Based on Genre and Cast", recommended_by_genre_and_cast, "total_similarity")
display_recommendations("📚 Recommended K-Dramas Based on Genre", recommended_by_genre, "genre_similarity")
display_recommendations("👥 Recommended K-Dramas Based on Cast", recommended_by_cast, "cast_similarity")
