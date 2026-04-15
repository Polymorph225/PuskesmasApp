# 🏥 PuskesmasApp — Ensemble AI Forecasting Dashboard

Dashboard analisis & peramalan cerdas data kunjungan Puskesmas dengan arsitektur **Ensemble Model: Prophet + XGBoost + SARIMA + Auto-Selection**.

> Dikembangkan untuk UPT Puskesmas Purwosari, Kabupaten Bojonegoro  
> Bagian dari Proyek Aktualisasi Latsar CPNS 2026

---

## ✨ Fitur Utama

### 🆕 Upgrade Model (v2.0)
| Fitur | Sebelum (v1) | Sesudah (v2) |
|---|---|---|
| Model forecasting | Prophet saja | **Ensemble: Prophet + XGBoost + SARIMA** |
| Feature engineering | ❌ Tidak ada | ✅ Lag, rolling mean, musim hujan, dll |
| Evaluasi akurasi | ❌ Tidak ada | ✅ MAE, RMSE, MAPE ditampilkan ke pengguna |
| Deteksi pola diagnosa musiman | ❌ Tidak ada | ✅ Heatmap + komparasi Hujan vs Kemarau |
| Auto-selection model terbaik | ❌ Tidak ada | ✅ MAPE-based weighted ensemble |
| Granularitas data | Mingguan saja | ✅ Mingguan atau Bulanan |

### 📊 Semua Fitur
1. **Dashboard Ringkasan** — Tren kunjungan, distribusi poli, KPI utama
2. **Analisis Kunjungan** — Demografi gender & kelompok umur
3. **Analisis Penyakit** — Top diagnosa, grafik interaktif horizontal/vertikal
4. **🌧️ Penyakit per Musim (BARU)** — Heatmap bulanan, komparasi musim hujan vs kemarau, tren penyakit spesifik
5. **🗺️ Peta Persebaran** — Geospasial interaktif per desa
6. **🤖 Ensemble Forecasting (BARU)** — Prophet + XGBoost + SARIMA dengan evaluasi akurasi lengkap
7. **💳 Analisis Pembiayaan** — Distribusi jenis pembiayaan
8. **🧹 Kualitas Data** — Missing values, duplikasi
9. **💬 Asisten AI Gemini** — Konsultasi berbasis data real-time

---

## 🏗️ Arsitektur Model Ensemble

```
Data Input (6–12 bulan, agregat mingguan/bulanan)
        │
        ▼
┌───────────────────────────────────────────────┐
│           FEATURE ENGINEERING                 │
│  • Temporal: week, month, quarter, year       │
│  • Musim hujan Indonesia (Nov–Apr)            │
│  • Lag features: lag_1, lag_2, lag_3, lag_4  │
│  • Rolling mean: roll_mean_4, roll_mean_8     │
│  • Rolling std: roll_std_4                    │
└───────────────┬───────────────────────────────┘
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
   ┌─────────┐ ┌───────┐ ┌──────────────────┐
   │ PROPHET │ │  XGB  │ │ SARIMA(1,1,1)    │
   │         │ │ Boost │ │ (1,1,0)[52]      │
   │Regressors│ │300est.│ │ Seasonal weekly  │
   │musim    │ │LR=0.05│ │                  │
   └────┬────┘ └───┬───┘ └────────┬─────────┘
        │          │              │
        └──────────┼──────────────┘
                   ▼
        ┌─────────────────────┐
        │   AUTO-EVALUATION   │
        │  MAE, RMSE, MAPE    │
        │  per model          │
        └──────────┬──────────┘
                   │
        ┌──────────┼──────────┐
        ▼                     ▼
  Best Single Model    Weighted Ensemble
  (MAPE terendah)      (bobot = 1/MAPE)
        │                     │
        └──────────┬──────────┘
                   ▼
            Final Forecast
       (ditampilkan ke pengguna)
```

### Mengapa Ensemble?
- **Prophet** unggul menangkap tren jangka panjang dan pola tahunan
- **XGBoost** menangkap pola non-linear dari fitur lag & musim
- **SARIMA** kuat untuk komponen autoregresif musiman periodik
- **Weighted Ensemble**: model dengan error rendah mendapat bobot lebih tinggi → akurasi optimal

---

## 🚀 Instalasi & Menjalankan

### Prasyarat
- Python 3.10 atau lebih baru
- pip

### 1. Clone repositori
```bash
git clone https://github.com/Polymorph225/PuskesmasApp.git
cd PuskesmasApp
```

### 2. Buat virtual environment (disarankan)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Catatan untuk Windows:** Jika instalasi `prophet` gagal, install terlebih dahulu:
> ```bash
> pip install pystan==3.7.0
> pip install prophet
> ```

### 4. Konfigurasi API Key Gemini
Buat file `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "ISI_API_KEY_GOOGLE_ANDA"
```
*(Jangan pernah commit file ini ke GitHub)*

### 5. Jalankan aplikasi
```bash
streamlit run Puskesmas_app.py
```
Buka browser di `http://localhost:8501`

---

## 📂 Struktur Repositori

```
PuskesmasApp/
│
├── Puskesmas_app.py          # 🔑 Kode utama aplikasi Streamlit
├── requirements.txt          # Daftar library Python
├── README.md                 # Dokumentasi ini
│
├── Data Dummy 10000.xlsx     # Contoh data uji (10.000 baris)
├── Data Dummy 5000.xlsx      # Contoh data uji (5.000 baris)
│
└── .streamlit/               # (Lokal, jangan di-commit)
    └── secrets.toml          # API key rahasia
```

---

## 📋 Format Data yang Didukung

File CSV atau Excel dengan kolom-kolom berikut (nama fleksibel, sistem auto-detect):

| Kolom Standar | Alias yang Dikenali |
|---|---|
| `tanggal_kunjungan` | `tgl_kunjungan`, `tanggal`, `visit_date` |
| `no_rm` | `no_rekam_medis`, `norm`, `rekammedis` |
| `umur` | `usia`, `age` |
| `jenis_kelamin` | `jk`, `sex`, `kelamin` |
| `poli` | `unit`, `unit_layanan`, `poliklinik` |
| `diagnosa` | `diagnosis`, `icd10`, `diag` |
| `pembiayaan` | `cara_bayar`, `penjamin`, `jaminan` |
| `desa` | `kelurahan`, `alamat_desa` |

---

## 🎯 Cara Penggunaan Ensemble Forecasting

1. Upload file data di sidebar
2. Atur filter sesuai kebutuhan (opsional)
3. Klik menu **"🤖 Ensemble Forecasting"**
4. Pilih jenis analisis: **Diagnosa Penyakit** atau **Poli/Unit**
5. Pilih item spesifik (misal: "ISPA", "Poli Umum")
6. Tentukan tanggal target prediksi dan frekuensi (Mingguan/Bulanan)
7. Klik **"🚀 Jalankan Ensemble Forecasting"**
8. Lihat:
   - Tabel evaluasi akurasi (MAE, RMSE, MAPE) per model
   - Grafik perbandingan semua model + ensemble
   - Ringkasan estimasi total kunjungan

---

## 📈 Interpretasi Metrik Akurasi

| MAPE | Interpretasi |
|---|---|
| < 10% | ✅ Sangat Baik |
| 10–20% | 🟡 Baik |
| 20–30% | 🟠 Cukup |
| > 30% | 🔴 Perlu Perbaikan (data kurang / pola tidak stabil) |

> **Tips:** Dengan data 6–12 bulan, MAPE 10–25% adalah hasil yang wajar untuk data kunjungan Puskesmas yang memiliki variasi tinggi.

---

## 🛠️ Teknologi

| Library | Fungsi |
|---|---|
| Streamlit | Framework UI web interaktif |
| Prophet | Time-series forecasting (tren + seasonality) |
| XGBoost | Gradient boosting dengan feature engineering |
| Statsmodels (SARIMA) | Seasonal ARIMA untuk pola periodik |
| scikit-learn | Evaluasi metrik, TimeSeriesSplit |
| Plotly | Visualisasi interaktif & heatmap |
| Pandas / NumPy | Manipulasi data |
| Google Generative AI | Asisten AI (Gemini) |
| FPDF2 | Ekspor laporan PDF |

---

## 🔮 Roadmap Pengembangan Lanjutan

- [ ] Integrasi data curah hujan BMKG sebagai fitur eksogen
- [ ] Deteksi otomatis outbreak berbasis Z-score / kontrol chart
- [ ] Model per poli (multi-output forecasting)
- [ ] Notifikasi otomatis jika prediksi melebihi threshold
- [ ] Export laporan PDF termasuk grafik ensemble

---

## 📞 Kontak

Dikembangkan oleh peserta Latsar CPNS 2026  
UPT Puskesmas Purwosari, Kabupaten Bojonegoro

Untuk pertanyaan atau masukan, buka **Issue** di repositori ini.

---

## 📄 Lisensi

MIT License — lihat file [LICENSE](LICENSE)
