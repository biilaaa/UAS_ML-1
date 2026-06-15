# ⚡ KLASIFIKASI INSTALASI KABEL PADA TIANG LISTRIK MENGGUNAKAN COMPUTER VISION SEBAGAI UPAYA MITIGASI BAHAYA

## Deskripsi Proyek

Proyek ini merupakan implementasi sistem klasifikasi citra tiang listrik berbasis Deep Learning menggunakan arsitektur MobileNetV2 dan Transfer Learning. Sistem dikembangkan untuk membantu proses identifikasi kondisi tiang listrik secara otomatis berdasarkan citra digital yang diunggah oleh pengguna.

Aplikasi mampu mengklasifikasikan kondisi tiang listrik ke dalam dua kategori utama, yaitu:

* **AMAN (Tiang Layak / Kabel Rapi)**
* **BAHAYA (Tiang Berisiko / Kabel Semrawut)**

Untuk mempermudah penggunaan, model diimplementasikan ke dalam aplikasi web interaktif menggunakan framework Streamlit sehingga pengguna dapat melakukan pengujian secara langsung melalui browser.

---

## Latar Belakang

Inspeksi kondisi tiang listrik secara manual membutuhkan waktu, tenaga, dan biaya yang tidak sedikit. Selain itu, proses inspeksi manual rentan terhadap subjektivitas penilaian.

Dengan memanfaatkan teknologi Deep Learning dan Computer Vision, proses identifikasi kondisi tiang listrik dapat dilakukan secara otomatis melalui analisis citra. Pendekatan ini diharapkan mampu membantu petugas dalam melakukan monitoring infrastruktur kelistrikan secara lebih cepat, konsisten, dan efisien.

---

## Tujuan Proyek

Tujuan dari proyek ini adalah:

1. Membangun model klasifikasi citra berbasis Deep Learning untuk mendeteksi kondisi tiang listrik.
2. Mengimplementasikan arsitektur MobileNetV2 menggunakan metode Transfer Learning.
3. Mengembangkan aplikasi web yang mudah digunakan untuk melakukan prediksi kondisi tiang listrik.
4. Membantu proses inspeksi visual infrastruktur kelistrikan secara otomatis.

---

## Teknologi yang Digunakan

### Bahasa Pemrograman

* Python

### Framework dan Library

* TensorFlow
* Keras
* Streamlit
* NumPy
* Pillow (PIL)

### Arsitektur Deep Learning

* MobileNetV2
* Transfer Learning

---

## Struktur Folder Proyek

```text
Project/
│
├── app-mobilenet.py
├── model_tiang_mobilenetv2.h5
├── requirements.txt
├── README.md
└── dataset/
```

---

## Arsitektur Model

Model dibangun menggunakan arsitektur MobileNetV2 yang telah dilatih sebelumnya (pre-trained model) dan dimanfaatkan melalui pendekatan Transfer Learning.

Keunggulan MobileNetV2:

* Ringan dan efisien
* Cocok untuk perangkat dengan sumber daya terbatas
* Memiliki performa tinggi pada tugas klasifikasi citra
* Waktu inferensi relatif cepat
* Jumlah parameter lebih sedikit dibandingkan model CNN konvensional

Pada tahap klasifikasi, MobileNetV2 digunakan sebagai feature extractor untuk mengekstraksi karakteristik visual dari citra tiang listrik sebelum dilakukan proses klasifikasi akhir.

---

## Dataset

Dataset yang digunakan terdiri dari citra tiang listrik yang telah diberi label berdasarkan kondisi visualnya.

Kategori kelas:

### Kelas 0 – AMAN

* Tiang listrik dalam kondisi layak
* Kabel tertata dengan rapi
* Tidak ditemukan indikasi risiko yang signifikan

### Kelas 1 – BAHAYA

* Kabel semrawut
* Instalasi kabel tidak rapi
* Terdapat indikasi kondisi yang memerlukan perhatian atau perbaikan

---

## Tahapan Pengolahan Data

### 1. Input Citra

Pengguna mengunggah gambar dengan format:

* JPG
* JPEG
* PNG

### 2. Resize Citra

Seluruh gambar diubah ukurannya menjadi:

```python
224 x 224 pixel
```

sesuai dengan ukuran input MobileNetV2.

### 3. Preprocessing

Citra diproses menggunakan:

```python
preprocess_input()
```

untuk menyesuaikan format data dengan kebutuhan MobileNetV2.

### 4. Prediksi

Model melakukan inferensi terhadap citra dan menghasilkan probabilitas untuk masing-masing kelas.

### 5. Output

Sistem menampilkan:

* Status klasifikasi
* Tingkat keyakinan (confidence score)

---

## Hasil Evaluasi Model

Model dievaluasi menggunakan beberapa metrik klasifikasi untuk mengukur performa pada data pengujian.

| No | Metrik Evaluasi | Nilai  |
| -- | --------------- | ------ |
| 1  | Accuracy        | 96.50% |
| 2  | Precision       | 96.73% |
| 3  | Recall          | 96.50% |
| 4  | F1-Score        | 96.50% |

---

## Interpretasi Hasil

### Accuracy – 96.50%

Menunjukkan bahwa model mampu melakukan klasifikasi dengan benar pada 96.50% data pengujian.

### Precision – 96.73%

Menunjukkan bahwa sebagian besar prediksi yang diberikan model merupakan prediksi yang benar sehingga tingkat kesalahan klasifikasi relatif rendah.

### Recall – 96.50%

Menunjukkan kemampuan model dalam menemukan dan mengidentifikasi seluruh data yang benar-benar termasuk dalam suatu kelas.

### F1-Score – 96.50%

Menunjukkan keseimbangan yang sangat baik antara Precision dan Recall sehingga model memiliki performa yang stabil dan konsisten.

---

## Kesimpulan Performa Model

Berdasarkan hasil evaluasi, model MobileNetV2 berhasil mencapai tingkat akurasi sebesar 96.50%. Nilai Precision, Recall, dan F1-Score yang tinggi menunjukkan bahwa model memiliki kemampuan klasifikasi yang sangat baik dalam membedakan kondisi tiang listrik yang aman maupun berpotensi bahaya.

Hasil ini menunjukkan bahwa model dapat digunakan sebagai alat bantu inspeksi visual infrastruktur kelistrikan secara otomatis dan efisien.

---

## Cara Menjalankan Program

### 1. Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependensi

```bash
pip install streamlit tensorflow numpy pillow
```

atau

```bash
pip install -r requirements.txt
```

### 3. Pastikan File Model Tersedia

Simpan file berikut pada folder proyek:

```text
model_tiang_mobilenetv2.h5
```

### 4. Jalankan Aplikasi

```bash
streamlit run app-mobilenet.py
```

### 5. Buka Browser

Aplikasi akan berjalan pada:

```text
http://localhost:8501
```

---

## Cara Menggunakan Aplikasi

1. Jalankan aplikasi Streamlit.
2. Klik tombol Upload File.
3. Pilih gambar tiang listrik.
4. Tunggu proses prediksi selesai.
5. Sistem akan menampilkan:

   * Hasil klasifikasi
   * Tingkat keyakinan model

---

## Contoh Hasil Prediksi

### Output Kelas Aman

```text
STATUS KELAYAKAN

AMAN
(Tiang Layak / Kabel Rapi)

Tingkat Keyakinan:
98.42%
```

### Output Kelas Berpotensi Bahaya

```text
STATUS KELAYAKAN

BERPOTENSI BAHAYA
(Tiang Berisiko / Kabel Semrawut)

Tingkat Keyakinan:
97.15%
```

---

## Kelebihan Sistem

* Antarmuka sederhana dan mudah digunakan
* Berbasis web menggunakan Streamlit
* Waktu prediksi cepat
* Akurasi tinggi (96.50%)
* Memanfaatkan Transfer Learning MobileNetV2
* Dapat digunakan sebagai alat bantu inspeksi visual

---

## Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:

* Menambahkan dataset dengan variasi kondisi lingkungan yang lebih banyak.
* Menambahkan fitur deteksi objek untuk menunjukkan area kabel yang bermasalah.
* Implementasi pada perangkat mobile.
* Integrasi dengan sistem monitoring infrastruktur kelistrikan.
* Deployment ke cloud agar dapat diakses secara online.

---

## Author

**Zahra Nabila**
Program Studi Teknologi Informasi
Universitas Lambung Mangkurat

---

## License

Proyek ini dibuat untuk keperluan pembelajaran dan pengembangan sistem klasifikasi citra berbasis Deep Learning.
