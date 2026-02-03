import streamlit as st
import pandas as pd
import numpy as np
import os
import re

from google import genai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. PENGATURAN KONFIGURASI
# ==========================================
st.set_page_config(
    page_title="Dashboard Analisis Data Puskesmas (Standar Kemenkes)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. CSS & TAMPILAN
# ==========================================
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Global background */
        body { background-color: #f5f7fb; }
        .block-container {
            padding-top: 1.5rem; padding-bottom: 3rem;
            padding-left: 2rem; padding-right: 2rem;
        }
        h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
        h1 { font-weight: 700 !important; color: #2c3e50; }
        
        /* Kartu Metric */
        div[data-testid="metric-container"] {
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #f8fafc; }
        
        /* Tabel */
        div[data-testid="stDataFrame"] {
            border-radius: 0.5rem; border: 1px solid #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

# Header Utama
st.title("🏥 Dashboard Puskesmas & SPM Monitor")
st.markdown(
    """
    Dashboard analisis data pelayanan kesehatan berbasis standar **Kemenkes RI**.
    Fitur utama:
    - 📊 **Analisis Kunjungan & Penyakit** (ICD-10)
    - 📋 **Monitoring SPM** (Standar Pelayanan Minimal: HT, DM, TB, dll)
    - 🤖 **Asisten AI** untuk rekomendasi kebijakan kesehatan
    """
)

# ==========================================
# 3. KONEKSI GENAI (GEMINI)
# ==========================================
@st.cache_resource
def get_gemini_client():
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        pass

    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error init Gemini: {e}")
        return None

# ==========================================
# 4. DATA CLEANING & PREPROCESSING (STANDAR KEMENKES)
# ==========================================

TARGET_COLS = [
    "tanggal_kunjungan", "no_rm", "nik", "nama", "umur", 
    "jenis_kelamin", "poli", "diagnosa", "pembiayaan", "desa"
]

def _normalize_col_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]", "", col)
    return col

def _build_column_mapping(df_raw: pd.DataFrame) -> dict:
    mapping = {}
    alias_dict = {
        "tanggal_kunjungan": ["tanggalkunjungan", "tglkunjungan", "tanggal", "tgl", "visitdate"],
        "no_rm": ["norm", "norekammedis", "no_rekam_medis", "mrn"],
        "nik": ["nik", "no_ktp", "nomorinduk", "nomor_induk_kependudukan"],
        "nama": ["nama", "namapasien", "namalengkap"],
        "umur": ["umur", "usia", "age", "umurth"],
        "jenis_kelamin": ["jeniskelamin", "jk", "sex", "kelamin", "gender"],
        "poli": ["poli", "unit", "unitlayanan", "ruangan"],
        "diagnosa": ["diagnosa", "dx", "icd10", "diagnosautama", "penyakit"],
        "pembiayaan": ["pembiayaan", "carabayar", "penjamin", "asuransi", "jenis_pasien"],
        "desa": ["desa", "kelurahan", "alamatdesa", "namadesa"],
    }
    
    norm_cols = {col: _normalize_col_name(col) for col in df_raw.columns}
    for std_name, alias_list in alias_dict.items():
        for raw_col, norm_key in norm_cols.items():
            if norm_key in alias_list:
                mapping[raw_col] = std_name
                break
    return mapping

def _clean_value_jenis_kelamin(val):
    if pd.isna(val): return np.nan
    v = str(val).strip().lower()
    if v in ["l", "lk", "laki", "laki-laki", "male", "1"]: return "Laki-Laki"
    if v in ["p", "pr", "perempuan", "wanita", "female", "2"]: return "Perempuan"
    return str(val).title()

def _parse_umur(val):
    if pd.isna(val): return np.nan
    s = str(val)
    m = re.search(r"(\d+)", s)
    if m:
        try: return int(m.group(1))
        except: return np.nan
    try: return int(float(s))
    except: return np.nan

def clean_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty: return df_raw
    
    col_map = _build_column_mapping(df_raw)
    df = df_raw.rename(columns=col_map).copy()
    
    # Ambil kolom yang ada di TARGET_COLS saja
    keep_cols = [c for c in TARGET_COLS if c in df.columns]
    df = df[keep_cols].copy()
    
    # Basic Cleaning
    if "tanggal_kunjungan" in df.columns:
        df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
    if "umur" in df.columns:
        df["umur"] = df["umur"].apply(_parse_umur)
    if "jenis_kelamin" in df.columns:
        df["jenis_kelamin"] = df["jenis_kelamin"].apply(_clean_value_jenis_kelamin)
    
    # String standar
    for col in ["poli", "diagnosa", "pembiayaan", "desa", "no_rm", "nik", "nama"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "NaT": np.nan})
            
    return df

@st.cache_data
def load_data(file):
    filename = file.name.lower()
    ext = filename.split(".")[-1]
    df_raw = None
    try:
        if ext == "csv":
            try: df_raw = pd.read_csv(file)
            except: 
                file.seek(0)
                df_raw = pd.read_csv(file, sep=";") # Coba separator lain
        elif ext in ["xlsx", "xls"]:
            df_raw = pd.read_excel(file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None, None
        
    df_clean = clean_raw_data(df_raw)
    return df_clean, df_raw

def preprocess_data(df):
    """
    Enrichment data dengan standar Kemenkes:
    1. Kategori Umur Depkes
    2. Labeling SPM (Hipertensi, DM, dll)
    """
    if df is None or df.empty: return df

    # Tanggal
    if "tanggal_kunjungan" in df.columns:
        df["tahun"] = df["tanggal_kunjungan"].dt.year
        df["bulan"] = df["tanggal_kunjungan"].dt.month
        df["nama_bulan"] = df["tanggal_kunjungan"].dt.strftime("%b")

    # --- STANDAR KEMENKES: KATEGORI UMUR (DEPKES RI 2009) ---
    if "umur" in df.columns:
        df["umur"] = pd.to_numeric(df["umur"], errors="coerce")
        # Bins: 0-5 (Balita), 5-11 (Kanak2), 12-25 (Remaja), 26-45 (Dewasa), 46-65 (Lansia), >65 (Manula)
        # Disederhanakan untuk visualisasi dashboard
        bins = [0, 5, 12, 26, 46, 65, 200]
        labels = ["Balita (0-5)", "Anak (6-11)", "Remaja (12-25)", "Dewasa (26-45)", "Lansia (46-65)", "Manula (>65)"]
        df["kategori_umur_depkes"] = pd.cut(df["umur"], bins=bins, labels=labels, right=False)
        
        # Kelompok Umur Lama (backup)
        df["kelompok_umur"] = df["kategori_umur_depkes"] 

    # --- STANDAR KEMENKES: INDIKATOR SPM ---
    if "diagnosa" in df.columns:
        def deteksi_spm(dx):
            dx = str(dx).upper()
            spm = []
            # Kode ICD-10 Umum / Keyword
            if "I10" in dx or "HIPERTENSI" in dx or "DARAH TINGGI" in dx: spm.append("SPM Hipertensi")
            if any(x in dx for x in ["E11", "E10", "DIABETES", "DM TIPE"]): spm.append("SPM Diabetes")
            if any(x in dx for x in ["A15", "TUBERKULOSIS", "TBC", "BTA"]): spm.append("SPM TB")
            if any(x in dx for x in ["F20", "SKIZOFRENIA", "ODGJ", "GANGGUAN JIWA"]): spm.append("SPM ODGJ")
            if any(x in dx for x in ["O80", "PERSALINAN", "HAMIL"]): spm.append("SPM Ibu Hamil")
            
            return ", ".join(spm) if spm else "Non-SPM"
            
        df["label_spm"] = df["diagnosa"].apply(deteksi_spm)

    # Standarisasi Teks Judul
    for col in ["poli", "jenis_kelamin", "pembiayaan", "desa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.title()
            
    return df

# ==========================================
# 5. FILTER & SIDEBAR
# ==========================================
def apply_filters(_):
    with st.sidebar:
        st.header("📂 Upload Data Puskesmas")
        uploaded_file = st.file_uploader("Format: CSV / Excel", type=["csv", "xlsx", "xls"])
        
        st.info(
            "🔒 **Privasi (UU PDP):** Data diproses di browser/lokal. "
            "Pastikan menghapus nama lengkap jika tidak diperlukan untuk analisis."
        )

        if uploaded_file is None:
            st.warning("Silakan upload data kunjungan.")
            return None, None

        df_clean, df_raw = load_data(uploaded_file)
        df = preprocess_data(df_clean)
        
        st.success(f"✅ Data dimuat: {len(df):,} baris")
        
        # --- FILTER CONTROL ---
        st.markdown("### 🔍 Filter Dashboard")
        
        # Tanggal
        date_range = None
        if "tanggal_kunjungan" in df.columns:
            min_d, max_d = df["tanggal_kunjungan"].min(), df["tanggal_kunjungan"].max()
            if pd.notna(min_d):
                date_range = st.date_input("Rentang Tanggal", [min_d, max_d])

        # Poli
        poli_pilihan = None
        if "poli" in df.columns:
            opts = sorted(df["poli"].dropna().unique())
            poli_pilihan = st.multiselect("Poli / Unit", opts, default=opts)
            
        # SPM Filter (Baru)
        spm_pilihan = None
        if "label_spm" in df.columns:
            opts_spm = sorted(df["label_spm"].dropna().unique())
            spm_pilihan = st.multiselect("Status SPM", opts_spm, default=opts_spm)

    # --- APPLY FILTER LOGIC ---
    df_filtered = df.copy()
    
    if date_range and len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered["tanggal_kunjungan"].dt.date >= date_range[0]) & 
            (df_filtered["tanggal_kunjungan"].dt.date <= date_range[1])
        ]
        
    if poli_pilihan:
        df_filtered = df_filtered[df_filtered["poli"].isin(poli_pilihan)]
        
    if spm_pilihan:
        df_filtered = df_filtered[df_filtered["label_spm"].isin(spm_pilihan)]

    filter_info = {"poli": poli_pilihan, "spm": spm_pilihan}
    return df_filtered, filter_info

def show_active_filters(info):
    if not info: return
    # Hanya helper visual simpel
    st.caption("✅ Filter aktif diterapkan pada data.")

# ==========================================
# 6. HALAMAN - HALAMAN DASHBOARD
# ==========================================

def page_overview(df, info):
    st.subheader("📌 Ringkasan Eksekutif Puskesmas")
    show_active_filters(info)
    
    if df.empty:
        st.warning("Data kosong.")
        return
        
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Kunjungan", f"{len(df):,}")
    if "no_rm" in df.columns:
        c2.metric("Pasien Unik", f"{df['no_rm'].nunique():,}")
    if "label_spm" in df.columns:
        cnt_spm = len(df[df["label_spm"] != "Non-SPM"])
        c3.metric("Kunjungan SPM", f"{cnt_spm:,}", help="Hipertensi, DM, TB, ODGJ, dll")
    if "poli" in df.columns:
        c4.metric("Poli Aktif", df["poli"].nunique())
        
    st.markdown("---")
    
    # Grafik Tren
    col_chart, col_pie = st.columns([2, 1])
    
    with col_chart:
        st.markdown("##### 📈 Tren Kunjungan Bulanan")
        if "bulan" in df.columns:
            trend = df.groupby(["tahun", "bulan", "nama_bulan"]).size().reset_index(name="count").sort_values(["tahun", "bulan"])
            trend["periode"] = trend["nama_bulan"] + "-" + trend["tahun"].astype(str)
            st.line_chart(trend.set_index("periode")["count"])
            
    with col_pie:
        st.markdown("##### 🏥 Top 5 Poli")
        if "poli" in df.columns:
            top_poli = df["poli"].value_counts().head(5)
            st.dataframe(top_poli, use_container_width=True)

def page_spm_monitor(df, info):
    st.subheader("📋 Monitoring Standar Pelayanan Minimal (SPM)")
    st.markdown("Memantau indikator kinerja wajib Puskesmas (HT, DM, TB, ODGJ, Ibu Hamil).")
    
    if "label_spm" not in df.columns or df.empty:
        st.warning("Data SPM tidak terdeteksi.")
        return

    # Filter hanya data SPM
    df_spm = df[df["label_spm"] != "Non-SPM"].copy()
    
    if df_spm.empty:
        st.info("Tidak ada kunjungan yang tergolong indikator SPM pada filter saat ini.")
        return

    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("#### Distribusi Kasus SPM")
        spm_counts = df_spm["label_spm"].value_counts()
        st.bar_chart(spm_counts)
        st.dataframe(spm_counts.rename("Jumlah Pasien"))
        
    with c2:
        st.markdown("#### Detail Pasien SPM (By Name/Address)")
        st.caption("Daftar ini untuk petugas P-Care / Perkesmas melakukan tindak lanjut.")
        
        cols_show = [c for c in ["tanggal_kunjungan", "no_rm", "nik", "nama", "diagnosa", "label_spm", "desa"] if c in df.columns]
        st.dataframe(df_spm[cols_show].head(50), hide_index=True, use_container_width=True)
        
        # Download Button
        csv = df_spm[cols_show].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Laporan SPM (Excel/CSV)",
            data=csv,
            file_name="laporan_spm_puskesmas.csv",
            mime="text/csv"
        )

def page_kunjungan(df, info):
    st.subheader("👥 Analisis Demografi Pasien")
    
    c1, c2 = st.columns(2)
    
    with c1:
        if "kategori_umur_depkes" in df.columns:
            st.markdown("#### Kategori Umur (Standar Depkes)")
            umur_counts = df["kategori_umur_depkes"].value_counts().sort_index()
            st.bar_chart(umur_counts)
            
    with c2:
        if "jenis_kelamin" in df.columns:
            st.markdown("#### Jenis Kelamin")
            jk_counts = df["jenis_kelamin"].value_counts()
            st.bar_chart(jk_counts, color="#ffaa00")

    if "desa" in df.columns:
        st.markdown("---")
        st.markdown("#### 🗺️ Sebaran Wilayah (Desa/Kelurahan)")
        desa_counts = df["desa"].value_counts().head(15)
        st.bar_chart(desa_counts, horizontal=True)

def page_penyakit(df, info):
    st.subheader("🦠 Analisis Diagnosa (ICD-10)")
    
    if "diagnosa" not in df.columns:
        st.error("Kolom diagnosa tidak ditemukan.")
        return
        
    top_n = st.slider("Jumlah Diagnosa Terbanyak", 5, 20, 10)
    
    top_diag = df["diagnosa"].value_counts().head(top_n)
    st.bar_chart(top_diag)
    
    with st.expander("Lihat Crosstab Diagnosa x Kelompok Umur"):
        if "kategori_umur_depkes" in df.columns:
            # Ambil top 10 diagnosa saja agar tabel rapi
            top10 = df["diagnosa"].value_counts().head(10).index
            df_top = df[df["diagnosa"].isin(top10)]
            ctab = pd.crosstab(df_top["diagnosa"], df_top["kategori_umur_depkes"])
            st.dataframe(ctab.style.background_gradient(cmap="Blues"))

def page_ml_prediction(df, info):
    st.subheader("🧠 Prediksi Risiko Lonjakan Kasus")
    st.info("Menggunakan Random Forest untuk memprediksi tren kunjungan poli/diagnosa.")
    
    if "tanggal_kunjungan" not in df.columns or df.empty:
        return
        
    target = st.selectbox("Pilih Target Prediksi", ["poli", "diagnosa"], index=0)
    
    if target not in df.columns: return
    
    # Ambil item terpopuler
    top_items = df[target].value_counts().head(20).index
    pilihan = st.selectbox(f"Pilih {target}", top_items)
    
    # Proses Data Time Series
    df_item = df[df[target] == pilihan].copy()
    df_item["period"] = df_item["tanggal_kunjungan"].dt.to_period("M")
    ts = df_item.groupby("period").size().reset_index(name="count")
    ts["period_dt"] = ts["period"].dt.to_timestamp()
    
    st.line_chart(ts.set_index("period_dt")["count"])
    
    if len(ts) < 6:
        st.warning("Data historis kurang dari 6 bulan, prediksi tidak akurat.")
        return
        
    # Simple Feature Engineering
    ts["month"] = ts["period_dt"].dt.month
    ts["year"] = ts["period_dt"].dt.year
    ts["t"] = np.arange(len(ts))
    
    # Threshold Lonjakan (misal > percentile 75)
    threshold = ts["count"].quantile(0.75)
    ts["is_spike"] = (ts["count"] >= threshold).astype(int)
    
    # Train Model
    X = ts[["t", "month", "year"]]
    y = ts["is_spike"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict Future (6 bulan)
    last_t = ts["t"].max()
    last_date = ts["period_dt"].max()
    
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    future_df = pd.DataFrame({
        "period_dt": future_dates,
        "t": np.arange(last_t + 1, last_t + 7),
        "month": future_dates.month,
        "year": future_dates.year
    })
    
    probs = model.predict_proba(future_df[["t", "month", "year"]])[:, 1]
    future_df["prob_lonjakan"] = probs
    
    st.markdown("#### 🔮 Probabilitas Lonjakan 6 Bulan Kedepan")
    st.line_chart(future_df.set_index("period_dt")["prob_lonjakan"])
    
    high_risk = future_df[future_df["prob_lonjakan"] > 0.6]
    if not high_risk.empty:
        st.error(f"⚠️ Peringatan: Potensi lonjakan kasus pada bulan: {', '.join(high_risk['period_dt'].dt.strftime('%B %Y'))}")

def page_ai_assistant(df, client):
    st.subheader("🤖 Asisten Analis Kesehatan (AI)")
    
    if not client:
        st.error("API Key Gemini belum diset. Tambahkan di `st.secrets` atau environment variable.")
        return

    st.markdown("Tanyakan interpretasi data, saran program promkes, atau analisis kesenjangan SPM.")
    
    # Ringkasan Data untuk Konteks AI
    summary = []
    summary.append(f"Total Kunjungan: {len(df)}")
    summary.append(f"Top 3 Poli: {', '.join(df['poli'].value_counts().head(3).index.astype(str))}")
    summary.append(f"Top 3 Diagnosa: {', '.join(df['diagnosa'].value_counts().head(3).index.astype(str))}")
    if "label_spm" in df.columns:
         spm_stats = df["label_spm"].value_counts().to_dict()
         summary.append(f"Statistik SPM: {spm_stats}")
         
    context = "\n".join(summary)
    
    user_q = st.text_area("Pertanyaan Anda:", height=100, placeholder="Contoh: Bagaimana cara meningkatkan capaian SPM Hipertensi berdasarkan data ini?")
    
    if st.button("Tanya AI"):
        if not user_q:
            st.warning("Tulis pertanyaan dulu.")
            return
            
        prompt = f"""
        Anda adalah konsultan kesehatan masyarakat ahli data Puskesmas di Indonesia.
        Berikut ringkasan data aktual Puskesmas:
        {context}
        
        Pertanyaan User: {user_q}
        
        Berikan jawaban yang taktis, sesuai regulasi Kemenkes RI, dan fokus pada solusi praktis (Promotif/Preventif/Kuratif).
        """
        
        with st.spinner("Sedang menganalisis..."):
            try:
                resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                st.markdown(resp.text)
            except Exception as e:
                st.error(f"Gagal koneksi AI: {e}")

# ==========================================
# 7. MAIN APP ROUTING
# ==========================================
def main():
    # Sidebar Navigation
    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Menu Navigasi", 
        ["Ringkasan Umum", "Monitoring SPM", "Analisis Kunjungan", "Analisis Penyakit", "Prediksi ML", "Asisten AI"]
    )
    
    # Load & Filter Data
    df_filtered, filter_info = apply_filters(None)
    client = get_gemini_client()
    
    # Routing
    if df_filtered is not None:
        if menu == "Ringkasan Umum":
            page_overview(df_filtered, filter_info)
        elif menu == "Monitoring SPM":
            page_spm_monitor(df_filtered, filter_info)
        elif menu == "Analisis Kunjungan":
            page_kunjungan(df_filtered, filter_info)
        elif menu == "Analisis Penyakit":
            page_penyakit(df_filtered, filter_info)
        elif menu == "Prediksi ML":
            page_ml_prediction(df_filtered, filter_info)
        elif menu == "Asisten AI":
            page_ai_assistant(df_filtered, client)
    else:
        # Tampilan Awal jika belum upload
        st.write("👈 Silakan upload file data kunjungan (Excel/CSV) di sidebar sebelah kiri.")

if __name__ == "__main__":
    main()
