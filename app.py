import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Prediksi Tingkat Bunuh Diri di Amerika Serikat")

# Memuat model
model = joblib.load('model.pkl')

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi", "Deskripsi Model", "Tentang"])

# Halaman Prediksi
if page == "Prediksi":
    st.header("Prediksi Tingkat Bunuh Diri")
    # Input tahun dari pengguna
    year = st.number_input('Masukkan Tahun:', min_value=1950, max_value=2024, step=1)
    age_group = st.selectbox('Pilih Kelompok Usia:', ['All ages', '0-14', '15-24', '25-34', '35-54', '55-74', '75+'])
    age_num = 0  # Default untuk "All ages", sesuaikan dengan dataset kamu
    
    # Prediksi
    if st.button('Prediksi'):
        prediction = model.predict([[year, age_num]])[0]
        st.write(f"Prediksi Tingkat Bunuh Diri per 100,000 penduduk pada tahun {year} untuk kelompok usia {age_group} adalah {prediction:.2f}")

# Halaman Deskripsi Model
elif page == "Deskripsi Model":
    st.header("Deskripsi Model")
    st.write("""
    Model yang digunakan adalah regresi linear sederhana untuk memprediksi tingkat bunuh diri berdasarkan tahun dan kelompok usia.
    Data yang digunakan berasal dari dataset tingkat bunuh diri di Amerika Serikat.
    """)

# Halaman Tentang
elif page == "Tentang":
    st.header("Tentang Aplikasi Ini")
    st.write("""
    Aplikasi ini dibuat untuk memprediksi tingkat bunuh diri di Amerika Serikat berdasarkan tahun dan kelompok usia.
    Aplikasi ini menggunakan model regresi linear yang dilatih dengan dataset historis.
    """)
