import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import plotly.express as px
import plotly.graph_objects as go

# Library Machine Learning & AI
from google import genai
from prophet import Prophet

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="Dashboard Analisis Data Puskesmas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== TAMPILAN (CSS) ADAPTIF ==========
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* CSS Umum */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1 { font-weight: 700 !important; }
        
        div[data-testid="metric-container"] {
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            background-color: var(--secondary-background-color); 
            border: 1px solid var(--faded-text-color);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
        }
        
        div[data-testid="stDataFrame"] {
            border-radius: 0.5rem;
            border: 1px solid var(--faded-text-color);
        }
        
        hr { margin: 0.75rem 0 1rem 0; }
        
        /* ==============================================
           KELAS KHUSUS UNTUK TEKS ESTIMASI PROPHET
           ============================================== */
        .highlight-estimasi {
            color: #1d4ed8 !important; /* Selalu warna Biru Tua/Tegas di mode terang maupun gelap */
            line-height: 1.5;
            font-size: 1.05rem;
            font-weight: 800; /* Ditebalkan maksimal agar stroke lebih terlihat jelas */
            
            /* Efek Stroke Warna Putih di sekeliling huruf */
            text-shadow: 
                -1px -1px 0 #ffffff,  
                 1px -1px 0 #ffffff,
                -1px  1px 0 #ffffff,
                 1px  1px 0 #ffffff,
                 0px  0px  5px rgba(255,255,255,0.8); /* Tambahan pendaran cahaya putih di belakangnya */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

# ========== HEADER UTAMA ==========
st.title("📊 Dashboard Analisis Data Puskesmas")

st.markdown(
    """
    Dashboard ini membantu menganalisis **data kunjungan Puskesmas** berdasarkan:
    - Poli / unit layanan  
    - Diagnosa  
    - Jenis kelamin & kelompok umur  
    - Jenis pembiayaan  
    - Wilayah pelayanan (desa/kelurahan)  

    ⬅️ **Mulai dengan meng-upload file data di sidebar, lalu atur filter sesuai kebutuhan.**
    """
)

# ========= FUNGSI UTAMA PERSIAPAN DATA DAN AI =========

@st.cache_resource
def get_gemini_client():
    """
    Membuat client Gemini berdasarkan API key dari:
    1. st.secrets["GEMINI_API_KEY"], atau
    2. environment variable GEMINI_API_KEY / GOOGLE_API_KEY
    """
    api_key = None

    # 1. mencoba ambil dari st.secrets (lebih aman untuk produksi)
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None

    # 2. kalau belum ada, coba dari environment variable
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        st.warning(
            "API key Gemini belum diset. "
            "Set di st.secrets['GEMINI_API_KEY'] atau environment variable GEMINI_API_KEY."
        )
        return None

    client = genai.Client(api_key=api_key)
    return client

# ================== DATA CLEANING HELPER ==================

TARGET_COLS = ["tanggal_kunjungan", "no_rm", "umur", "jenis_kelamin", "poli", "diagnosa", "pembiayaan", "desa"]

def _normalize_col_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]", "", col)
    return col

def _build_column_mapping(df_raw: pd.DataFrame) -> dict:
    mapping = {}
    alias_dict = {
        "tanggal_kunjungan": ["tanggalkunjungan", "tglkunjungan", "tanggal", "tgl", "visitdate"],
        "no_rm": ["norm", "norekammedis", "norekam_medis", "no_rekam_medis", "rekammedis"],
        "umur": ["umur", "usia", "age", "umurth"],
        "jenis_kelamin": ["jeniskelamin", "jk", "sex", "kelamin", "llp"],
        "poli": ["poli", "politujuan", "unit", "unitlayanan", "poliklinik"],
        "diagnosa": ["diagnosa", "diagnosautama", "dxutama", "diag", "icd10", "diagnosis"],
        "pembiayaan": ["pembiayaan", "carabayar", "penjamin", "jaminan", "pembayar"],
        "desa": ["desa", "alamatdesa", "kelurahan", "desakelurahan", "namadesa"],
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
    if v in ["l", "lk", "laki", "laki-laki", "male", "m", "1"]: return "Laki-Laki"
    if v in ["p", "pr", "perempuan", "wanita", "female", "f", "2"]: return "Perempuan"
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
    keep_cols = [c for c in TARGET_COLS if c in df.columns]
    df = df[keep_cols].copy()
    
    if "tanggal_kunjungan" in df.columns:
        df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
    if "umur" in df.columns:
        df["umur"] = df["umur"].apply(_parse_umur)
    if "jenis_kelamin" in df.columns:
        df["jenis_kelamin"] = df["jenis_kelamin"].apply(_clean_value_jenis_kelamin)
    
    for col in ["poli", "diagnosa", "pembiayaan", "desa", "no_rm"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
    return df

@st.cache_data
def load_data(file):
    filename = file.name.lower()
    ext = filename.split(".")[-1]
    df_raw = None
    if ext == "csv":
        df_raw = pd.read_csv(file)
    elif ext in ["xlsx", "xls"]:
        try: df_raw = pd.read_excel(file, engine="openpyxl")
        except: df_raw = pd.read_excel(file, engine="xlrd")
    
    df_clean = clean_raw_data(df_raw)
    return df_clean, df_raw

def preprocess_data(df):
    if df is None or df.empty: return df
    df.columns = [c.strip().lower() for c in df.columns]
    
    if "desa" not in df.columns:
        for c in df.columns:
            if "desa" in c.lower() and "alamat" not in c.lower():
                df["desa"] = df[c]
                break

    if "umur" in df.columns:
        df["umur"] = pd.to_numeric(df["umur"], errors="coerce")
        bins = [0, 1, 5, 15, 25, 45, 60, 200]
        labels = ["<1 th", "1-4 th", "5-14 th", "15-24 th", "25-44 th", "45-59 th", "60+ th"]
        df["kelompok_umur"] = pd.cut(df["umur"], bins=bins, labels=labels, right=False)
        
    if "tanggal_kunjungan" in df.columns:
        df["tahun"] = df["tanggal_kunjungan"].dt.year
        df["bulan"] = df["tanggal_kunjungan"].dt.month
        df["nama_bulan"] = df["tanggal_kunjungan"].dt.strftime("%b")
        
    for col in ["poli", "jenis_kelamin", "pembiayaan", "diagnosa", "desa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

def apply_filters(_):
    with st.sidebar:
        st.markdown("## 🏥 Data Kunjungan Puskesmas")
        uploaded_file = st.file_uploader("Upload data kunjungan (CSV / Excel)", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is None:
            st.info("Silakan upload file **CSV/Excel**.")
            return None, None

        df_clean, df_raw = load_data(uploaded_file)
        df = preprocess_data(df_clean)
        
        st.success("Data berhasil dimuat ✅")
        st.caption(f"📊 {len(df):,} baris • {len(df.columns)} kolom".replace(",", "."))

        # Filter Variables
        date_range = None
        tahun_pilihan = poli_pilihan = jk_pilihan = bayar_pilihan = kelompok_umur_pilihan = desa_pilihan = None
        
        st.markdown("### 🔍 Filter Data")
        
        with st.expander("Filter Waktu & Poli", expanded=True):
            if "tanggal_kunjungan" in df.columns:
                min_d, max_d = df["tanggal_kunjungan"].min(), df["tanggal_kunjungan"].max()
                if pd.notna(min_d) and pd.notna(max_d):
                    date_range = st.date_input("Rentang Tanggal", value=[min_d.date(), max_d.date()])
            if "tahun" in df.columns:
                tahun_pilihan = st.multiselect("Tahun", options=sorted(df["tahun"].dropna().unique()))
            if "poli" in df.columns:
                poli_pilihan = st.multiselect("Poli", options=sorted(df["poli"].dropna().unique()))

        with st.expander("Filter Lainnya", expanded=False):
            if "jenis_kelamin" in df.columns:
                jk_pilihan = st.multiselect("Jenis Kelamin", options=sorted(df["jenis_kelamin"].dropna().unique()))
            if "kelompok_umur" in df.columns:
                kelompok_umur_pilihan = st.multiselect("Kelompok Umur", options=df["kelompok_umur"].dropna().unique())
            if "desa" in df.columns:
                desa_pilihan = st.multiselect("Desa", options=sorted(df["desa"].dropna().unique()))
            if "pembiayaan" in df.columns:
                bayar_pilihan = st.multiselect("Pembiayaan", options=sorted(df["pembiayaan"].dropna().unique()))

    # Apply Logic
    df_filtered = df.copy()
    if date_range and len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered["tanggal_kunjungan"].dt.date >= date_range[0]) & 
                                  (df_filtered["tanggal_kunjungan"].dt.date <= date_range[1])]
    if tahun_pilihan: df_filtered = df_filtered[df_filtered["tahun"].isin(tahun_pilihan)]
    if poli_pilihan: df_filtered = df_filtered[df_filtered["poli"].isin(poli_pilihan)]
    if jk_pilihan: df_filtered = df_filtered[df_filtered["jenis_kelamin"].isin(jk_pilihan)]
    if bayar_pilihan: df_filtered = df_filtered[df_filtered["pembiayaan"].isin(bayar_pilihan)]
    if kelompok_umur_pilihan: df_filtered = df_filtered[df_filtered["kelompok_umur"].isin(kelompok_umur_pilihan)]
    if desa_pilihan: df_filtered = df_filtered[df_filtered["desa"].isin(desa_pilihan)]
    
    filter_info = {
        "poli": poli_pilihan, "jenis_kelamin": jk_pilihan, "pembiayaan": bayar_pilihan,
        "kelompok_umur": kelompok_umur_pilihan, "desa": desa_pilihan
    }
    return df_filtered, filter_info

def show_active_filters(filter_info):
    if not filter_info: return
    chips = []
    for k, v in filter_info.items():
        if v: chips.append(f"{k.title()}: {', '.join(map(str, v))}")
    if chips: st.caption("🎯 **Filter aktif:** " + " | ".join(chips))

# ========= HALAMAN DASHBOARD =========

def page_overview(df_filtered, filter_info):
    st.subheader("📌 Ringkasan Umum")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Kunjungan", len(df_filtered))
    if "no_rm" in df_filtered.columns: col2.metric("Pasien Unik", df_filtered["no_rm"].nunique())
    if "poli" in df_filtered.columns: col3.metric("Poli Aktif", df_filtered["poli"].nunique())
    if "diagnosa" in df_filtered.columns: col4.metric("Diagnosa", df_filtered["diagnosa"].nunique())
    
    st.markdown("---")
    if "tanggal_kunjungan" in df_filtered.columns:
        st.markdown("### 📈 Tren Kunjungan")
        trend = df_filtered.groupby(["tahun", "bulan", "nama_bulan"]).size().reset_index(name="count").sort_values(["tahun", "bulan"])
        trend["label"] = trend["nama_bulan"].astype(str) + "-" + trend["tahun"].astype(str)
        st.line_chart(trend.set_index("label")["count"])

    if "poli" in df_filtered.columns:
        st.markdown("### 🏥 Distribusi Poli")
        st.bar_chart(df_filtered["poli"].value_counts())

def page_kunjungan(df_filtered, filter_info):
    st.subheader("👥 Analisis Kunjungan")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    
    col1, col2 = st.columns(2)
    if "jenis_kelamin" in df_filtered.columns:
        col1.markdown("#### Jenis Kelamin")
        col1.bar_chart(df_filtered["jenis_kelamin"].value_counts())
    if "kelompok_umur" in df_filtered.columns:
        col2.markdown("#### Kelompok Umur")
        col2.bar_chart(df_filtered["kelompok_umur"].value_counts().sort_index())

def page_penyakit(df_filtered, filter_info):
    st.subheader("🦠 Analisis Penyakit")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    if "diagnosa" in df_filtered.columns:
        top_n = st.slider("Jumlah diagnosa", 5, 20, 10)
        st.bar_chart(df_filtered["diagnosa"].value_counts().head(top_n))

# ================== HALAMAN PETA ==================
def page_peta_persebaran(df_filtered, filter_info):
    st.subheader("🗺️ Peta Persebaran Penyakit")
    show_active_filters(filter_info)
    
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong.")
        return
        
    if "desa" not in df_filtered.columns or "diagnosa" not in df_filtered.columns:
        st.error("❌ Kolom 'desa' atau 'diagnosa' tidak ditemukan dalam data.")
        return

    st.markdown("### 🔍 Filter Persebaran")
    
    col_pilih_penyakit, col_pilih_tema = st.columns([3, 1])
    with col_pilih_penyakit:
        top_penyakit = df_filtered["diagnosa"].value_counts().head(20).index.tolist()
        pilihan_penyakit = st.selectbox("Pilihan Diagnosa:", options=["-- Semua Penyakit (Top 10) --"] + top_penyakit)
    with col_pilih_tema:
        tema_peta = st.selectbox("Tema Peta:", ["Terang", "Gelap"])

    map_style = "carto-darkmatter" if tema_peta == "Gelap" else "carto-positron"

    if pilihan_penyakit != "-- Semua Penyakit (Top 10) --":
        df_map = df_filtered[df_filtered["diagnosa"] == pilihan_penyakit].copy()
    else:
        top_10 = df_filtered["diagnosa"].value_counts().head(10).index.tolist()
        df_map = df_filtered[df_filtered["diagnosa"].isin(top_10)].copy()

    df_grouped = df_map.groupby(["desa", "diagnosa"]).size().reset_index(name="jumlah_kasus")

    # KOORDINAT DESA KECAMATAN PURWOSARI
    koordinat_desa = {
        "Donan": (-7.215046, 111.635988),
        "Gapluk": (-7.200003, 111.662517),
        "Kaliombo": (-7.243656, 111.686128),
        "Kuniran": (-7.223605, 111.655825),
        "Ngrejeng": (-7.222318, 111.706618),
        "Pelem": (-7.237117, 111.699898),
        "Pojok": (-7.237117, 111.699898),
        "Punggur": (-7.203616, 111.682271),
        "Purwosari": (-7.176333, 111.662109),
        "Sedahkidul": (-7.196343, 111.679325),
        "Tinumpuk": (-7.213089, 111.682381),
        "Tlatah": (-7.213808, 111.696004)
    }

    koordinat_default = (-7.1509, 111.8817)

    def get_koordinat(nama_desa):
        desa_bersih = str(nama_desa).strip().title()
        return koordinat_desa.get(desa_bersih, koordinat_default)

    df_grouped["latitude"] = df_grouped["desa"].apply(lambda x: get_koordinat(x)[0])
    df_grouped["longitude"] = df_grouped["desa"].apply(lambda x: get_koordinat(x)[1])

    st.markdown("---")
    total_kasus = df_grouped["jumlah_kasus"].sum()
    total_desa = df_grouped["desa"].nunique()
    
    desa_agregat = df_grouped.groupby("desa")["jumlah_kasus"].sum()
    desa_tertinggi = desa_agregat.idxmax() if not desa_agregat.empty else "-"
    kasus_tertinggi = desa_agregat.max() if not desa_agregat.empty else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Kasus Terpeta", value=f"{total_kasus} Pasien")
    with col2:
        st.metric(label="Total Desa Terdampak", value=f"{total_desa} Desa")
    with col3:
        st.metric(label="Desa Kasus Tertinggi", value=f"{desa_tertinggi}", delta=f"{kasus_tertinggi} Kasus", delta_color="off")

    st.markdown(f"**Menampilkan persebaran pasien untuk:** `{pilihan_penyakit}`")

    # RENDER PETA PLOTLY
    fig = px.scatter_mapbox(
        df_grouped, 
        lat="latitude", 
        lon="longitude", 
        color="diagnosa",
        size="jumlah_kasus",
        hover_name="desa",
        hover_data={"latitude": False, "longitude": False, "diagnosa": True, "jumlah_kasus": True},
        color_discrete_sequence=px.colors.qualitative.Plotly, 
        zoom=11.5, 
        center={"lat": -7.218, "lon": 111.675}, 
        height=550,
        size_max=35 
    )
    
    fig.update_layout(mapbox_style=map_style)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    
    col_tabel, col_grafik = st.columns([1, 1])
    with col_tabel:
        st.markdown("#### 📋 Detail Kasus per Desa")
        st.dataframe(df_grouped[["desa", "diagnosa", "jumlah_kasus"]].sort_values(by="jumlah_kasus", ascending=False), use_container_width=True, hide_index=True)

    with col_grafik:
        st.markdown("#### 📊 Top Desa Terdampak")
        top_desa_df = df_grouped.groupby("desa")["jumlah_kasus"].sum().reset_index().sort_values("jumlah_kasus", ascending=False).head(10)
        st.bar_chart(top_desa_df.set_index("desa")["jumlah_kasus"])

def page_pembiayaan(df_filtered, filter_info):
    st.subheader("💳 Analisis Pembiayaan")
    if df_filtered is None or "pembiayaan" not in df_filtered.columns: return
    st.bar_chart(df_filtered["pembiayaan"].value_counts())

# ================== HALAMAN ML (FACEBOOK PROPHET) ==================
def page_ml(df_filtered, filter_info):
    st.subheader("📈 Prediksi & Peramalan Tren (Facebook Prophet)")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong.")
        return

    if "tanggal_kunjungan" not in df_filtered.columns:
        st.error("❌ Kolom 'tanggal_kunjungan' tidak ditemukan.")
        return

    # Filter data ML
    df_ml = df_filtered.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_ml["tanggal_kunjungan"]):
        df_ml["tanggal_kunjungan"] = pd.to_datetime(df_ml["tanggal_kunjungan"], errors="coerce")
    df_ml = df_ml.dropna(subset=["tanggal_kunjungan"])
    
    if len(df_ml) == 0:
        st.warning("⚠️ Data tanggal tidak valid.")
        return

    max_date = df_ml["tanggal_kunjungan"].max().date()

    st.info("💡 **Model:** Menggunakan Facebook Prophet untuk memprediksi tren dan estimasi jumlah kunjungan/kasus di masa depan.")
    st.markdown("---")

    with st.container():
        col_set1, col_set2, col_set3 = st.columns([1, 1, 1])
        with col_set1:
            fokus = st.radio("Analisis:", ["Diagnosa Penyakit", "Poli / Unit"], horizontal=True)
            kolom_fokus = "diagnosa" if fokus == "Diagnosa Penyakit" else "poli"
        
        with col_set2:
            if kolom_fokus not in df_ml.columns: return
            top_items = df_ml[kolom_fokus].value_counts().head(30)
            pilihan_item = st.selectbox(f"Pilih {fokus}:", options=top_items.index.tolist())
            
        with col_set3:
            target_date = st.date_input(
                "Prediksi Sampai Tanggal:", 
                value=max_date + pd.Timedelta(days=30), 
                min_value=max_date + pd.Timedelta(days=1),
                max_value=max_date + pd.Timedelta(days=365) # Maks 1 tahun
            )

    df_item = df_ml[df_ml[kolom_fokus] == pilihan_item].copy()
    
    if len(df_item) < 10:
        st.error("❌ Data terlalu sedikit (< 10 pasien) untuk dipelajari oleh AI Prophet.")
        return

    # ================= LOGIKA PROPHET =================
    weekly = df_item.groupby(pd.Grouper(key="tanggal_kunjungan", freq="W-MON")).size().reset_index(name="jumlah")
    df_prophet = weekly.rename(columns={"tanggal_kunjungan": "ds", "jumlah": "y"})

    with st.spinner('AI Prophet sedang memproses pola peramalan...'):
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.80)
        model.fit(df_prophet)

        last_date = df_prophet["ds"].max().date()
        target_dt = pd.to_datetime(target_date).date()
        delta_days = (target_dt - last_date).days
        periods_ahead = max(1, delta_days // 7)

        future = model.make_future_dataframe(periods=periods_ahead, freq='W-MON')
        forecast = model.predict(future)

    st.markdown(f"### 📈 Grafik Peramalan: **{pilihan_item}**")
    fig = go.Figure()

    # Data Aktual
    fig.add_trace(go.Scatter(
        x=df_prophet['ds'], y=df_prophet['y'], 
        mode='lines+markers', name='Data Aktual',
        line=dict(color='#2563eb', width=2)
    ))

    # Prediksi Masa Depan
    forecast_future = forecast[forecast['ds'] > pd.to_datetime(last_date)]
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'], y=forecast_future['yhat'], 
        mode='lines+markers', name='Prediksi Masa Depan',
        line=dict(color='#ef4444', width=3, dash='dash')
    ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'].tolist() + forecast_future['ds'].tolist()[::-1],
        y=forecast_future['yhat_upper'].tolist() + forecast_future['yhat_lower'].tolist()[::-1],
        fill='toself', fillcolor='rgba(239, 68, 68, 0.2)', 
        line=dict(color='rgba(255,255,255,0)'),
        name='Rentang Toleransi',
        hoverinfo="skip"
    ))

    fig.update_layout(
        xaxis_title="Periode Waktu", yaxis_title="Jumlah Kunjungan/Pasien",
        hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ================= KESIMPULAN ESTIMASI DENGAN CUSTOM HTML/CSS =================
    st.markdown(f"### 📢 Kesimpulan Estimasi Hingga {pd.to_datetime(target_date).strftime('%d %B %Y')}")
    if not forecast_future.empty:
        total_estimasi = int(round(forecast_future["yhat"].clip(lower=0).sum()))
        
        target_week = forecast_future.iloc[-1]
        tgl_target_akhir = target_week["ds"].strftime("%d %B %Y")
        
        est_kunjungan_akhir = max(0, int(round(target_week["yhat"])))
        batas_bawah_akhir = max(0, int(round(target_week["yhat_lower"])))
        batas_atas_akhir = max(0, int(round(target_week["yhat_upper"])))

        col_alert, col_metric1, col_metric2 = st.columns([2, 1, 1])
        with col_alert:
            # Menggunakan CSS class `.highlight-estimasi` (warna biru tua, tebal, stroke tepi putih)
            st.markdown(f"""
                <div style="padding: 1rem; border-radius: 0.5rem; background-color: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); margin-bottom: 1rem;">
                    <div class="highlight-estimasi">
                        📆 Selama periode ke depan hingga <b>{tgl_target_akhir}</b>, AI memperkirakan akan ada <b>total akumulasi {total_estimasi} kunjungan/kasus</b> untuk <b>{pilihan_item}</b>. <br><br>
                        Sementara itu, khusus pada pekan terakhir tersebut, diprediksi terdapat <b>{est_kunjungan_akhir} kunjungan</b> baru.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_metric1:
            st.metric("Total Akumulasi Pasien", f"{total_estimasi} Pasien")
        with col_metric2:
            st.metric("Estimasi Pekan Terakhir", f"{est_kunjungan_akhir} Pasien", help=f"Batas Bawah: {batas_bawah_akhir} | Batas Atas: {batas_atas_akhir}")
    else:
        st.info("Tanggal prediksi terlalu dekat dengan data terakhir.")

def page_data(df_filtered, filter_info):
    st.subheader("📄 Data & Unduhan")
    if df_filtered is None: return
    st.dataframe(df_filtered)
    st.download_button("💾 Download CSV", df_filtered.to_csv(index=False).encode("utf-8"), "data_puskesmas.csv", "text/csv")

def page_quality(df):
    st.subheader("🧹 Kualitas Data")
    if df is None: return
    st.write(f"Duplikasi: {df.duplicated().sum()} baris")
    st.write("Missing Values:")
    st.dataframe(df.isna().sum().to_frame("Missing Count"))

def page_ai_assistant(df_filtered, filter_info, client):
    st.subheader("🤖 Asisten AI")
    if df_filtered is None: return
    if not client: 
        st.error("API Key belum diset.")
        return

    user_q = st.text_area("Tanya AI tentang data ini:", placeholder="Saran program untuk diagnosa terbanyak?")
    if st.button("Kirim"):
        context = f"Total data: {len(df_filtered)}. Top Poli: {df_filtered['poli'].mode()[0] if 'poli' in df_filtered else '-'}"
        prompt = f"Sebagai analis kesehatan. Data: {context}. Pertanyaan: {user_q}"
        with st.spinner("Berpikir..."):
            try:
                res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                st.markdown(res.text)
            except Exception as e:
                error_msg = str(e)
                # Tangkapan error anti-panik untuk limit kuota Gemini API
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    st.warning("⏳ **Sistem AI sedang sibuk (Limit Penggunaan).** Silakan tunggu sekitar 1 menit, lalu klik 'Kirim' lagi.")
                else:
                    st.error("❌ Gagal terhubung ke server AI. Pastikan internet Anda stabil atau API Key valid.")

# ========= MAIN =========

def main():
    df_filtered, filter_info = apply_filters(None)
    client = get_gemini_client()

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigasi", [
        "Ringkasan Umum", "Analisis Kunjungan", "Analisis Penyakit", 
        "Peta Persebaran",
        "Analisis Pembiayaan", "Data & Unduhan", "Kualitas Data", 
        "Prediksi ML", "Asisten AI"
    ])
    
    if page == "Ringkasan Umum": page_overview(df_filtered, filter_info)
    elif page == "Analisis Kunjungan": page_kunjungan(df_filtered, filter_info)
    elif page == "Analisis Penyakit": page_penyakit(df_filtered, filter_info)
    elif page == "Peta Persebaran": page_peta_persebaran(df_filtered, filter_info)
    elif page == "Analisis Pembiayaan": page_pembiayaan(df_filtered, filter_info)
    elif page == "Data & Unduhan": page_data(df_filtered, filter_info)
    elif page == "Kualitas Data": page_quality(df_filtered)
    elif page == "Prediksi ML": page_ml(df_filtered, filter_info)
    elif page == "Asisten AI": page_ai_assistant(df_filtered, filter_info, client)

if __name__ == "__main__":
    main()

