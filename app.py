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

# Hitung similarity (contoh sederhana untuk demo)
df['similarity'] = df.apply(
    lambda x: cosine_similarity_manual(
        [1 if genre in drama_detail['Genre'] else 0 for genre in x['Genre']],
        [1] * len(drama_detail['Genre'])
    ),
    axis=1
)

# Hapus drama yang dipilih dari daftar rekomendasi
df = df[df['Name'] != selected_drama]

# Pilih 5 rekomendasi berdasarkan similarity tertinggi
recommended_dramas = df.sort_values(by='similarity', ascending=False).head(5)

# Tampilkan rekomendasi dalam layout card
st.subheader("Recommended K-Dramas for You")
for _, drama in recommended_dramas.iterrows():
    st.markdown(f"""
    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; margin-bottom:15px; box-shadow: 0px 2px 4px rgba(0,0,0,0.1);">
        <h4 style="margin:0;">{drama['Name']}</h4>
        <p style="margin:0;"><strong>Rating:</strong> {drama['Rating'] if 'Rating' in drama else 'N/A'}</p>
        <p style="margin:0;"><strong>Number of Episodes:</strong> {drama['Number of Episodes'] if 'Number of Episodes' in drama else 'N/A'}</p>
        <p style="margin:0;"><strong>Genre:</strong> {', '.join(drama['Genre'])}</p>
        <p style="margin:0;"><strong>Cast:</strong> {drama['Cast'] if 'Cast' in drama else 'N/A'}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("*Created with Streamlit* © 2025")
