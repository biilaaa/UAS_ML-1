import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# ==============================================================================
# 1. KONFIGURASI HALAMAN WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Tiang Listrik - MobileNetV2",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Sistem Klasifikasi Kelayakan Tiang Listrik")
st.write("Aplikasi pengujian citra tiang listrik menggunakan arsitektur **Deep Learning MobileNetV2 End-to-End**.")
st.markdown("---")

# ==============================================================================
# 2. INJEKSI PATCHING SEBELUM LOAD MODEL (ANTI BENTROK VERSI COLAB)
# ==============================================================================
@st.cache_resource
def muat_model_keras_aman():
    nama_model = 'model_tiang_mobilenetv2.h5'
    if os.path.exists(nama_model):
        try:
            # --- TRIK UTAMA ---
            # Kita intercept fungsi deserialization Keras secara global di memori
            from keras.src.layers.core.dense import Dense
            from keras.src.models.functional import Functional
            
            # Simpan fungsi asli bawaan Keras lokal
            dense_from_config_asli = Dense.from_config
            functional_from_config_asli = Functional.from_config
            
            # Buat fungsi modifikasi untuk membuang parameter biang kerok
            @classmethod
            def safe_dense_from_config(cls, config):
                config.pop('quantization_config', None) # Buang parameter dari Dense Layer
                return dense_from_config_asli(config)
                
            @classmethod
            def safe_functional_from_config(cls, config):
                # Bersihkan parameter tersembunyi di dalam blok arsitektur MobileNetV2
                if 'layers' in config:
                    for layer in config['layers']:
                        if 'config' in layer:
                            layer['config'].pop('quantization_config', None)
                return functional_from_config_asli(config)
            
            # Suntikkan fungsi aman kita ke dalam core Keras sebelum load_model berjalan
            Dense.from_config = safe_dense_from_config
            Functional.from_config = safe_functional_from_config
            
            # Lakukan pemuatan model secara normal
            model_terlatih = load_model(nama_model)
            
            # Kembalikan fungsi asli Keras setelah model sukses dimuat (Good Practice)
            Dense.from_config = dense_from_config_asli
            Functional.from_config = functional_from_config_asli
            
            return model_terlatih
            
        except Exception as e:
            # Jika struktur path internal Keras berbeda (misal versi lama sekali), pakai alternative load alternatif
            try:
                model_terlatih = load_model(nama_model, compile=False)
                return model_terlatih
            except Exception as e_inner:
                st.error(f"⚠️ Gagal memuat model karena masalah kompatibilitas versi: {e_inner}")
                return None
    else:
        return None

# Eksekusi pemuatan model
model = muat_model_keras_aman()

if model is None:
    st.error("❌ Berkas model 'model_tiang_mobilenetv2.h5' tidak ditemukan atau gagal didekripsi di laptop lokal!")
    st.stop()

# ==============================================================================
# 3. ANTARMUKA UNGGAH CITRA & PREDIKSI
# ==============================================================================
uploaded_file = st.file_uploader("Pilih atau Tarik Foto Tiang Listrik...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    kolom_kiri, kolom_kanan = st.columns([1, 1])
    
    with kolom_kiri:
        st.subheader("📷 Citra Masukan")
        gambar_mentah = Image.open(uploaded_file)
        st.image(gambar_mentah, use_container_width=True)
        
    with kolom_kanan:
        st.subheader("🔍 Hasil Analisis Deep Learning")
        
        with st.spinner('Model sedang menganalisis piksel citra...'):
            try:
                # --- LANGKAH 1: PRE-PROCESSING ---
                gambar_proses = gambar_mentah.resize((224, 224))
                x = img_to_array(gambar_proses)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                # --- LANGKAH 2: INFERENCE (PREDIKSI MODEL) ---
                prediksi_mentah = model.predict(x, verbose=0)[0]
                indeks_tertinggi = np.argmax(prediksi_mentah)
                
                CATEGORIES = ['AMAN (Tiang Layak / Kabel Rapi)', 'BERPOTENSI BAHAYA (Tiang Beresiko / Kabel Semrawut)']
                status_akhir = CATEGORIES[indeks_tertinggi]
                skor_persentase = prediksi_mentah[indeks_tertinggi] * 100
                
                # --- LANGKAH 3: TAMPILKAN OUTPUT SECARA VISUAL ---
                if indeks_tertinggi == 0:
                    st.success(f"**STATUS KELAYAKAN:**\n\n{status_akhir}")
                    st.metric(label="Tingkat Keyakinan Klasifikasi", value=f"{skor_persentase:.2f}%")
                else:
                    st.error(f"**STATUS KELAYAKAN:**\n\n{status_akhir}")
                    st.metric(label="Tingkat Keyakinan Klasifikasi", value=f"{skor_persentase:.2f}%")
                    
            except Exception as e:
                st.error(f"Gagal memproses gambar saat ekstraksi: {e}")

# ==============================================================================
# 4. CATATAN KAKI METODOLOGI
# ==============================================================================
st.markdown("---")
st.caption("Informasi Arsitektur: Modifikasi runtime berhasil menyaring parameter eksternal Keras secara aman.")