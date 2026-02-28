# 🏥 Dashboard Analisis Data Puskesmas

Aplikasi ini adalah **dashboard interaktif berbasis Streamlit** untuk menganalisis data kunjungan Puskesmas.  
Fokusnya adalah membantu tenaga kesehatan, pengelola program, dan pengambil kebijakan di tingkat fasilitas pelayanan dasar untuk:

- Memahami pola kunjungan pasien
- Mengidentifikasi penyakit terbanyak
- Melihat distribusi berdasarkan demografi dan wilayah
- Menganalisis pola pembiayaan
- Melakukan proyeksi sederhana risiko lonjakan kasus dengan model Machine Learning
- Mendapatkan rekomendasi program melalui Asisten AI berbasis Gemini

---

## ✨ Fitur Utama

- 📂 **Upload Data Kunjungan**
  - Mendukung file **CSV**, **XLS**, dan **XLSX**
  - Otomatis melakukan _data cleaning_ dan normalisasi nama kolom (tanggal, no RM, umur, jenis kelamin, poli, diagnosa, pembiayaan, desa)
  - Menangani variasi nama kolom yang umum dipakai di Puskesmas

- 🧹 **Pembersihan & Standarisasi Data**
  - Konversi tanggal kunjungan ke tipe `datetime`
  - Normalisasi usia (termasuk format teks, misal: `36 Thn 9 Bln 19 Hari`)
  - Normalisasi jenis kelamin (`L`, `P`, `LK`, `PR`, dsb.) menjadi `Laki-Laki` dan `Perempuan`
  - Otomatis menghapus kolom alamat yang tidak relevan dan hanya mempertahankan kolom berbasis **desa**

- 🎛️ **Filter Interaktif**
  - Rentang **tanggal kunjungan**
  - **Tahun** kunjungan
  - **Poli / unit layanan**
  - **Jenis kelamin**
  - **Kelompok umur** (misal: `<1 th`, `1–4 th`, `5–14 th`, dst.)
  - **Desa / kelurahan**
  - **Jenis pembiayaan** (JKN, umum, dll.)

- 📊 **Ringkasan & Visualisasi**
  - Ringkasan total kunjungan, pasien unik, jumlah poli aktif, dan variasi diagnosa
  - Tren kunjungan per bulan
  - Distribusi kunjungan per poli, jenis kelamin, kelompok umur
  - Top N diagnosa terbanyak
  - Crosstab antar variabel (misalnya poli × jenis kelamin, diagnosa × jenis kelamin, diagnosa × kelompok umur)

- 💳 **Analisis Pembiayaan**
  - Ringkasan kunjungan per jenis pembiayaan
  - Matriks poli × pembiayaan untuk melihat pola penggunaan layanan dan jaminan kesehatan

- 🧠 **Model Machine Learning untuk Risiko Lonjakan Kasus**
  - Menggunakan **XGBoost** untuk memodelkan risiko **lonjakan kasus** per bulan pada diagnosa/poli tertentu
  - Definisi lonjakan berbasis **persentil historis** (dapat diatur oleh pengguna)
  - Menampilkan:
    - Tren historis kunjungan per bulan
    - Probabilitas lonjakan di beberapa bulan ke depan
    - Ringkasan bulan dengan risiko tinggi (mis. ≥70%)

- 🤖 **Asisten AI (Gemini) untuk Interpretasi & Rekomendasi**
  - Membuat ringkasan numerik dari data yang sudah difilter
  - Mengirim ringkasan tersebut ke model **Gemini** (Google Generative AI)
  - Menghasilkan:
    - Interpretasi pola kunjungan / penyakit / pembiayaan
    - Identifikasi masalah kesehatan prioritas
    - Rekomendasi program, intervensi, dan kebijakan yang relevan dengan konteks Puskesmas di Indonesia

- 📥 **Ekspor Data Terfilter**
  - Mengunduh data yang sudah difilter dalam format **CSV**
  - Membuat _pivot table_ sederhana langsung dari antarmuka aplikasi

- 🎨 **Tampilan Modern dengan Tema Dark & Light**
  - Pilihan **tema Dark** dan **Light** yang dapat diubah dari sidebar
  - Desain UI dengan:
    - Kartu metrik
    - Hero section
    - Kontras warna yang nyaman untuk penggunaan lama
    - Layout yang rapi dan mudah dibaca

---

## 🛠️ Teknologi yang Digunakan

- **Python**
- **Streamlit** – untuk UI dashboard interaktif
- **Pandas & NumPy** – untuk pemrosesan dan analisis data
- **scikit-learn** – untuk model Machine Learning (XGBoost)
- **Google Generative AI (Gemini)** – untuk Asisten AI (analisis dan rekomendasi)
- **CSS kustom** – untuk tema dark/light dan efek tampilan yang lembut

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/username/nama-repo.git
cd nama-repo

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # di Linux/Mac
venv\Scripts\activate           # di Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key Gemini (opsi)
# Bisa melalui environment variable:
# Export GEMINI_API_KEY="API_KEY_ANDA"
# Atau lewat st.secrets di Streamlit Cloud

# 5. Jalankan aplikasi
streamlit run app.py
