# 🏥 PuskesmasApp - Dashboard Analisis & Peramalan Cerdas

PuskesmasApp adalah sebuah aplikasi web interaktif (*dashboard*) yang dirancang khusus untuk mempermudah fasilitas kesehatan dalam mengolah, menganalisis, dan memvisualisasikan data kunjungan pasien atau rekam medis elektronik (RME). 

Aplikasi ini dikembangkan sebagai wujud inovasi digitalisasi pelayanan kesehatan di lingkungan **UPT Puskesmas Purwosari, Kabupaten Bojonegoro**, dan merupakan bagian dari proyek aktualisasi **Pelatihan Dasar (Latsar) CPNS Tahun 2026**.

---

## ✨ Fitur Utama

1. 📊 **Dashboard Analisis Komprehensif**
   Menganalisis tren kunjungan pasien, distribusi demografi (usia & jenis kelamin), jenis pembiayaan, hingga diagnosa penyakit terbanyak secara *real-time*.
2. 🗺️ **Peta Persebaran Penyakit (Geospasial)**
   Memetakan lokasi asal pasien berdasarkan desa di wilayah Kecamatan Purwosari (dan sekitarnya) secara interaktif untuk melihat wilayah rawan/endemik suatu penyakit.
3. 📈 **Prediksi & Peramalan Tren (AI Forecasting)**
   Terintegrasi dengan algoritma **Facebook Prophet** untuk memprediksi dan meramalkan estimasi jumlah pasien atau kasus penyakit di masa depan (hingga 1 tahun ke depan) guna membantu perencanaan anggaran dan logistik obat.
4. 🤖 **Asisten AI Kesehatan**
   Terkoneksi dengan **Google Gemini AI** yang berfungsi sebagai asisten pintar bagi tenaga kesehatan untuk merancang program penyuluhan, edukasi masyarakat, atau konsultasi administratif.

---

## 🛠️ Teknologi yang Digunakan

Aplikasi ini dibangun menggunakan bahasa pemrograman **Python** dengan dukungan *library* berikut:
- **[Streamlit](https://streamlit.io/):** *Framework* utama untuk membangun antarmuka web interaktif.
- **[Pandas & NumPy](https://pandas.pydata.org/):** Manipulasi dan pembersihan data struktural.
- **[Plotly Express](https://plotly.com/python/):** Pembuatan grafik visual dan rendering peta interaktif.
- **[Prophet](https://facebook.github.io/prophet/):** Model *Machine Learning* untuk *Time-Series Forecasting*.
- **[Google GenAI](https://ai.google.dev/):** Integrasi *Large Language Model* (LLM) untuk fitur Asisten AI.

---

## 🚀 Cara Instalasi & Menjalankan di Komputer Lokal (Localhost)

Jika Anda ingin menjalankan aplikasi ini di komputer/laptop Anda sendiri, ikuti langkah-langkah berikut:

### 1. Clone Repositori
```bash
git clone [https://github.com/username_anda/nama_repositori.git](https://github.com/username_anda/nama_repositori.git)
cd nama_repositori

2. Buat Virtual Environment (Opsional namun disarankan)
```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
venv\Scripts\activate     # Untuk Windows

3. Install Dependencies
```bash
pip install -r requirements.txt

4. Konfigurasi API Key (Untuk Fitur Asisten AI)
```bash
a. Agar Asisten AI Gemini dapat berfungsi, Anda memerlukan API Key dari Google.
b. Buat folder tersembunyi bernama .streamlit di dalam direktori proyek.
c. Buat file secrets.toml di dalam folder tersebut.
d. Isi file dengan format berikut:
GEMINI_API_KEY = "MASUKKAN_API_KEY_GOOGLE_ANDA_DI_SINI"
(Catatan: Jangan pernah memublikasikan file secrets.toml ke GitHub publik).

5. Jalankan Aplikasi
```bash
streamlit run Puskesmas_app.py

Aplikasi akan otomatis terbuka di browser Anda pada alamat http://localhost:8501.

📂 Struktur Repositori
📁 nama_repositori/
│
├── Puskesmas_app.py      # Kode sumber utama aplikasi Streamlit
├── requirements.txt      # Daftar pustaka (library) Python yang dibutuhkan
├── README.md             # Dokumentasi repositori ini
└── .streamlit/           # (Lokal) Folder konfigurasi Streamlit
    └── secrets.toml      # (Lokal) Kunci API Rahasia

💡 Cara Penggunaan Singkat

    Buka aplikasi PuskesmasApp.

    Pada panel sebelah kiri (Sidebar), unggah file data kunjungan pasien Anda dalam format .CSV atau .XLSX.

    Gunakan filter yang tersedia (rentang waktu, poli, jenis kelamin, desa) untuk memfokuskan data.

    Navigasikan menu di sidebar untuk melihat Ringkasan, Peta Persebaran, Peramalan AI, hingga berkonsultasi dengan Asisten AI.

📞 Kontak & Pengembang

Dikembangkan oleh peserta Latsar CPNS Tahun 2026.
Untuk pertanyaan, masukan, atau kendala terkait aplikasi ini, silakan hubungi pengembang melalui repositori ini (buka Issue atau hubungi langsung secara internal di UPT Puskesmas Purwosari).
