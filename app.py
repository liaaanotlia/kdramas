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
st.title("K-Drama Recommendation")
st.markdown("Korean drama recommendation for you")

# Load dataset
df = load_data()

# Dropdown untuk memilih drama
selected_drama = st.selectbox(
    "Select a Korean Drama:",
    options=df['Name'].values
)

# Detail drama yang dipilih
drama_detail = df[df['Name'] == selected_drama].iloc[0]

st.write(f"**Name:** {drama_detail['Name']}")
st.write(f"**Year of Release:** {drama_detail['Year of release'] if 'Year of release' in drama_detail else 'Data not available'}")
st.write(f"**Number of Episodes:** {drama_detail['Number of Episodes'] if 'Number of Episodes' in drama_detail else 'Data not available'}")
st.write(f"**Rating:** {drama_detail['Rating'] if 'Rating' in drama_detail else 'Data not available'}")
st.write(f"**Genre:** {', '.join(drama_detail['Genre'])}")
st.write(f"**Cast:** {drama_detail['Cast'] if 'Cast' in drama_detail else 'Data not available'}")

# Menghitung vektor genre untuk setiap drama
def compute_genre_vector(drama_genre, all_genres):
    return [1 if genre in drama_genre else 0 for genre in all_genres]

# Semua genre unik
all_genres = list({genre for genres in df['Genre'] for genre in genres})

# Vektor genre drama yang dipilih
selected_genre_vector = compute_genre_vector(drama_detail['Genre'], all_genres)

# Menghitung cosine similarity untuk semua drama
df['similarity'] = df.apply(
    lambda x: cosine_similarity_manual(
        compute_genre_vector(x['Genre'], all_genres),
        selected_genre_vector
    ),
    axis=1
)

# Hapus drama yang dipilih dari daftar rekomendasi
df = df[df['Name'] != selected_drama]

# Pilih 5 rekomendasi berdasarkan similarity tertinggi
recommended_dramas = df.sort_values(by='similarity', ascending=False).head(5)

# Tampilkan rekomendasi dalam layout grid
st.subheader("Recommended K-Dramas for You")
cols = st.columns(5)  # Membuat 5 kolom untuk grid

for col, (_, drama) in zip(cols, recommended_dramas.iterrows()):
    with col:
        st.markdown(f"### {drama['Name']}")
        st.write(f"**Rating:** {drama['Rating'] if 'Rating' in drama else 'N/A'}")
        st.write(f"**Episodes:** {drama['Number of Episodes'] if 'Number of Episodes' in drama else 'N/A'}")
        st.write(f"**Genre:** {', '.join(drama['Genre'])}")
        st.write(f"**Cast:** {drama['Cast'] if 'Cast' in drama else 'N/A'}")
