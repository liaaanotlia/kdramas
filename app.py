import streamlit as st
import pandas as pd
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("kdrama_cleaned.csv")
    
    # Menghapus spasi di sekitar nama kolom
    df.columns = df.columns.str.strip()
    df['Genre'] = df['Genre'].apply(lambda x: x.split())  # Memisahkan genre berdasarkan spasi
    df['Cast'] = df['Cast'].apply(lambda x: x.split(', '))  # Memisahkan cast berdasarkan koma
    return df

# Encode genre dan cast ke dalam format one-hot
def encode_features(data, column):
    terms = set(term for sublist in data[column] for term in sublist)  # Semua istilah unik
    encoded_matrix = []
    
    for doc in data[column]:
        encoded_doc = [1 if term in doc else 0 for term in terms]  # One-hot encoding
        encoded_matrix.append(encoded_doc)
    
    return np.array(encoded_matrix), list(terms)

# Fungsi untuk menghitung cosine similarity secara manual
def cosine_similarity_manual(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)  # Dot product
    norm_a = np.linalg.norm(vec_a)  # Magnitude vektor A
    norm_b = np.linalg.norm(vec_b)  # Magnitude vektor B
    if norm_a == 0 or norm_b == 0:  # Mencegah pembagian dengan nol
        return 0
    return dot_product / (norm_a * norm_b)

# Judul Aplikasi
st.title("K-Drama Recommendation")
st.markdown("Korean drama recommendation for you using Cosine Similarity (Manual Calculation)")

# Load dataset
df = load_data()

# Membuat dropdown untuk memilih drama Korea
selected_drama = st.selectbox(
    "Select a Korean Drama:",
    options=df['Name'].values
)

# Menampilkan detail drama yang dipilih
drama_detail = df[df['Name'] == selected_drama].iloc[0]

st.write(f"*Name:* {drama_detail['Name']}")
st.write(f"*Year of Release:* {drama_detail['Year of release'] if 'Year of release' in drama_detail else 'Data not available'}")
st.write(f"*Number of Episodes:* {drama_detail['Number of Episodes'] if 'Number of Episodes' in drama_detail else 'Data not available'}")
st.write(f"*Duration:* {drama_detail['Duration'] if 'Duration' in drama_detail else 'Data not available'}")
st.write(f"*Content Rating:* {drama_detail['Content Rating'] if 'Content Rating' in drama_detail else 'Data not available'}")
st.write(f"*Rating:* {drama_detail['Rating'] if 'Rating' in drama_detail else 'Data not available'}")
st.write(f"*Genre:* {' '.join(drama_detail['Genre'])}")
st.write(f"*Cast:* {', '.join(drama_detail['Cast'])}")

# Encode genre dan cast
genre_encoded, genre_terms = encode_features(df, 'Genre')
cast_encoded, cast_terms = encode_features(df, 'Cast')

# Gabungkan genre dan cast menjadi satu vektor
combined_encoded = np.hstack((genre_encoded, cast_encoded))

# Ambil indeks drama yang dipilih
selected_index = df[df['Name'] == selected_drama].index[0]

# Hitung cosine similarity secara manual untuk setiap drama
similarity_scores = []
for i, vector in enumerate(combined_encoded):
    if i != selected_index:  # Hindari membandingkan dengan diri sendiri
        similarity = cosine_similarity_manual(combined_encoded[selected_index], vector)
        similarity_scores.append((i, similarity))

# Sort berdasarkan similarity score
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Tambahkan similarity ke dataframe
df['Similarity'] = 0
for idx, score in similarity_scores:
    df.at[idx, 'Similarity'] = score

# Hapus drama yang dipilih dari rekomendasi
df = df[df['Name'] != selected_drama]

# Tampilkan rekomendasi berdasarkan similarity
st.subheader("Recommended K-Dramas:")
recommended_dramas = df.sort_values(by='Similarity', ascending=False).head(5)
st.dataframe(recommended_dramas[['Name', 'Rating', 'Number of Episodes', 'Genre', 'Cast', 'Similarity']])

# Footer
st.markdown("*Created with Streamlit* © 2025")
