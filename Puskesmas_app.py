import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import io
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime
import tempfile

# Library Machine Learning & AI
import google.generativeai as genai
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
            color: #1d4ed8 !important; /* Warna Biru Tua/Tegas */
            line-height: 1.5;
            font-size: 1.05rem;
            font-weight: 800; /* Teks ditebalkan */
        }

        /* ==============================================
           STYLE UNTUK TOMBOL SORT PENYAKIT
           ============================================== */
        .sort-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 0.3rem;
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
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error accessing secrets: {e}")

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        st.warning("⚠️ API key Gemini belum diset. Fitur AI Agent tidak akan berfungsi. Pastikan Anda telah mengatur GEMINI_API_KEY.")
        return False

    try:
        genai.configure(api_key=api_key)
        return True 
    except Exception as e:
        st.error(f"Gagal inisialisasi Gemini API: {e}")
        return False

# ================== EKSPOR EXCEL HELPER ==================
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    processed_data = output.getvalue()
    return processed_data

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
        kecuali_penyakit_pilihan = None
        
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
            
            if "diagnosa" in df.columns:
                kecuali_penyakit_pilihan = st.multiselect(
                    "❌ Kecualikan Penyakit", 
                    options=sorted(df["diagnosa"].dropna().unique()),
                    help="Penyakit yang dipilih di sini TIDAK AKAN diikutkan dalam analisis."
                )

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
    
    if kecuali_penyakit_pilihan:
        df_filtered = df_filtered[~df_filtered["diagnosa"].isin(kecuali_penyakit_pilihan)]
    
    filter_info = {
        "poli": poli_pilihan, "jenis_kelamin": jk_pilihan, "pembiayaan": bayar_pilihan,
        "kelompok_umur": kelompok_umur_pilihan, "desa": desa_pilihan,
        "penyakit_dikecualikan": kecuali_penyakit_pilihan
    }
    return df_filtered, filter_info

def show_active_filters(filter_info):
    if not filter_info: return
    chips = []
    for k, v in filter_info.items():
        if v: 
            label_bersih = k.replace("_", " ").title()
            chips.append(f"{label_bersih}: {', '.join(map(str, v))}")
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
        st.download_button("📥 Download Tren Kunjungan (Excel)", convert_df_to_excel(trend), "tren_kunjungan.xlsx")

    if "poli" in df_filtered.columns:
        st.markdown("### 🏥 Distribusi Poli")
        df_poli = df_filtered["poli"].value_counts().reset_index()
        df_poli.columns = ["Poli", "Jumlah"]
        st.bar_chart(df_poli.set_index("Poli"))
        st.download_button("📥 Download Distribusi Poli (Excel)", convert_df_to_excel(df_poli), "distribusi_poli.xlsx")

def page_kunjungan(df_filtered, filter_info):
    st.subheader("👥 Analisis Kunjungan")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    
    col1, col2 = st.columns(2)
    if "jenis_kelamin" in df_filtered.columns:
        col1.markdown("#### Jenis Kelamin")
        df_jk = df_filtered["jenis_kelamin"].value_counts().reset_index()
        df_jk.columns = ["Jenis Kelamin", "Jumlah"]
        col1.bar_chart(df_jk.set_index("Jenis Kelamin"))
        col1.download_button("📥 Download Data Gender", convert_df_to_excel(df_jk), "kunjungan_gender.xlsx")
        
    if "kelompok_umur" in df_filtered.columns:
        col2.markdown("#### Kelompok Umur")
        df_umur = df_filtered["kelompok_umur"].value_counts().sort_index().reset_index()
        df_umur.columns = ["Kelompok Umur", "Jumlah"]
        col2.bar_chart(df_umur.set_index("Kelompok Umur"))
        col2.download_button("📥 Download Data Umur", convert_df_to_excel(df_umur), "kunjungan_umur.xlsx")

# ================== HALAMAN ANALISIS PENYAKIT (DIPERBARUI) ==================
def page_penyakit(df_filtered, filter_info):
    st.subheader("🦠 Analisis Penyakit")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return

    if "diagnosa" not in df_filtered.columns:
        st.error("❌ Kolom 'diagnosa' tidak ditemukan dalam data.")
        return

    # ── Baris kontrol: slider + urutan + tampilan ──────────────────────────
    col_slider, col_urutan, col_tampilan = st.columns([2, 2, 1])

    with col_slider:
        top_n = st.slider("Jumlah diagnosa ditampilkan", min_value=5, max_value=30, value=10, step=1)

    with col_urutan:
        urutan = st.radio(
            "Urutkan berdasarkan jumlah kasus:",
            options=["⬇️ Terbanyak → Tersedikit", "⬆️ Tersedikit → Terbanyak"],
            index=0,
            horizontal=True,
            key="radio_urutan_penyakit"
        )

    with col_tampilan:
        orientasi = st.selectbox(
            "Orientasi grafik",
            options=["Horizontal", "Vertikal"],
            index=0,
            key="select_orientasi_penyakit"
        )

    # ── Olah data ──────────────────────────────────────────────────────────
    ascending = urutan.startswith("⬆️")

    df_diag = (
        df_filtered["diagnosa"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    df_diag.columns = ["Diagnosa", "Jumlah Kasus"]
    df_diag = df_diag.sort_values("Jumlah Kasus", ascending=ascending).reset_index(drop=True)

    # Tambah kolom peringkat untuk label hover
    df_diag["Peringkat"] = df_diag["Jumlah Kasus"].rank(ascending=False, method="min").astype(int)

    # ── Warna gradasi sesuai arah urutan ──────────────────────────────────
    color_scale = "Blues" if not ascending else "Blues_r"

    # ── Render grafik ──────────────────────────────────────────────────────
    chart_height = max(350, top_n * 38)

    if orientasi == "Horizontal":
        # Bar horizontal — terbaik untuk nama diagnosa panjang
        category_order = "total ascending" if ascending else "total descending"

        fig = px.bar(
            df_diag,
            x="Jumlah Kasus",
            y="Diagnosa",
            orientation="h",
            text="Jumlah Kasus",
            color="Jumlah Kasus",
            color_continuous_scale=color_scale,
            custom_data=["Peringkat"],
            labels={"Jumlah Kasus": "Jumlah Kasus", "Diagnosa": "Diagnosa"},
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Jumlah Kasus: <b>%{x}</b><br>"
                "Peringkat: #%{customdata[0]}<extra></extra>"
            ),
        )
        fig.update_layout(
            yaxis=dict(categoryorder=category_order),
            coloraxis_showscale=False,
            margin=dict(l=0, r=70, t=10, b=10),
            height=chart_height,
            xaxis_title="Jumlah Kasus",
            yaxis_title=None,
        )

    else:
        # Bar vertikal — cocok jika nama diagnosa pendek / sedikit item
        category_order = "total descending" if not ascending else "total ascending"

        fig = px.bar(
            df_diag,
            x="Diagnosa",
            y="Jumlah Kasus",
            orientation="v",
            text="Jumlah Kasus",
            color="Jumlah Kasus",
            color_continuous_scale=color_scale,
            custom_data=["Peringkat"],
            labels={"Jumlah Kasus": "Jumlah Kasus", "Diagnosa": "Diagnosa"},
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Jumlah Kasus: <b>%{y}</b><br>"
                "Peringkat: #%{customdata[0]}<extra></extra>"
            ),
        )
        fig.update_layout(
            xaxis=dict(categoryorder=category_order, tickangle=-35),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=30, b=10),
            height=chart_height,
            xaxis_title=None,
            yaxis_title="Jumlah Kasus",
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── Statistik ringkas di bawah grafik ─────────────────────────────────
    st.markdown("---")
    total_kasus_tampil = df_diag["Jumlah Kasus"].sum()
    total_semua_kasus  = len(df_filtered)
    pct = (total_kasus_tampil / total_semua_kasus * 100) if total_semua_kasus > 0 else 0

    diagnosa_teratas = df_diag.sort_values("Jumlah Kasus", ascending=False).iloc[0]["Diagnosa"]
    kasus_teratas    = df_diag.sort_values("Jumlah Kasus", ascending=False).iloc[0]["Jumlah Kasus"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Kasus (ditampilkan)", f"{total_kasus_tampil:,}".replace(",", "."))
    m2.metric("% dari Seluruh Kunjungan",  f"{pct:.1f}%")
    m3.metric("Diagnosa Teratas",           diagnosa_teratas, delta=f"{kasus_teratas} kasus", delta_color="off")

    # ── Tabel detail ──────────────────────────────────────────────────────
    with st.expander("📋 Lihat Tabel Detail", expanded=False):
        df_tabel = df_diag[["Peringkat", "Diagnosa", "Jumlah Kasus"]].copy()
        df_tabel["% dari Total"] = (df_tabel["Jumlah Kasus"] / total_semua_kasus * 100).round(2).astype(str) + "%"
        st.dataframe(df_tabel, use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────────────
    df_download = df_diag[["Peringkat", "Diagnosa", "Jumlah Kasus"]].copy()
    st.download_button(
        label="📥 Download Data Top Penyakit (Excel)",
        data=convert_df_to_excel(df_download),
        file_name="top_penyakit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

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

    koordinat_desa = {
        "Donan": (-7.2131, 111.6364),
        "Gapluk": (-7.2017, 111.6617),
        "Kaliombo": (-7.235, 111.6817),
        "Kuniran": (-7.234645, 111.651014),
        "Ngrejeng": (-7.226040889505065, 111.70707293891402),
        "Pelem": (-7.2394, 111.7011),
        "Pojok": (-7.1892, 111.6728),
        "Punggur": (-7.2048, 111.6808),
        "Purwosari": (-7.1798, 111.6608),
        "Sedahkidul": (-7.1973, 111.6792),
        "Tinumpuk": (-7.2117, 111.68),
        "Tlatah": (-7.2172, 111.6975)
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

    fig = px.scatter_mapbox(
        df_grouped, 
        lat="latitude", 
        lon="longitude", 
        color="diagnosa",
        size="jumlah_kasus",
        hover_name="desa",
        custom_data=["desa"], 
        hover_data={"latitude": False, "longitude": False, "diagnosa": True, "jumlah_kasus": True, "desa": False},
        color_discrete_sequence=px.colors.qualitative.Plotly, 
        zoom=11.5, 
        center={"lat": -7.218, "lon": 111.675}, 
        height=550,
        size_max=35 
    )
    
    fig.update_layout(mapbox_style=map_style)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    clicked_desa = None
    try:
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        if hasattr(event, "selection") and hasattr(event.selection, "points") and len(event.selection.points) > 0:
            point = event.selection.points[0]
            if "customdata" in point:
                clicked_desa = point["customdata"][0]
            elif "hovertext" in point:
                clicked_desa = point["hovertext"]
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)
        pilihan_fallback = st.selectbox("Pilih Desa untuk melihat statistik spesifik:", options=["-- Klik Pilih --"] + sorted(df_filtered["desa"].dropna().unique().tolist()))
        if pilihan_fallback != "-- Klik Pilih --":
            clicked_desa = pilihan_fallback

    st.info("👆 **TIPS INTERAKTIF:** Klik pada salah satu titik/lingkaran desa di peta untuk melihat profil statistik kesehatannya secara langsung!")

    if clicked_desa:
        st.markdown("---")
        st.markdown(f"### 📍 Profil Kesehatan Desa: **{clicked_desa}**")
        
        df_desa = df_filtered[df_filtered["desa"] == clicked_desa]
        
        if len(df_desa) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Kunjungan", f"{len(df_desa)} Pasien")
            
            penyakit_terbanyak = df_desa["diagnosa"].mode()[0] if "diagnosa" in df_desa.columns and not df_desa["diagnosa"].empty else "-"
            c2.metric("Penyakit Dominan", penyakit_terbanyak)
            
            if "jenis_kelamin" in df_desa.columns:
                mayoritas_gender = df_desa["jenis_kelamin"].mode()[0] if not df_desa["jenis_kelamin"].empty else "-"
                c3.metric("Mayoritas Gender", mayoritas_gender)
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Top 5 Penyakit Terbanyak**")
                st.bar_chart(df_desa["diagnosa"].value_counts().head(5))
            with col_d2:
                st.markdown("**Demografi Usia Pasien**")
                if "kelompok_umur" in df_desa.columns:
                    st.bar_chart(df_desa["kelompok_umur"].value_counts().sort_index())
                    
        else:
            st.warning(f"Data tidak tersedia untuk desa {clicked_desa}.")
            
        st.caption("ℹ️ *Klik pada area kosong di peta atau klik ulang titik desa untuk menutup profil ini.*")
        
    st.markdown("---")
    col_tabel, col_grafik = st.columns([1, 1])
    
    with col_tabel:
        st.markdown("#### 📋 Detail Kasus per Desa")
        st.dataframe(df_grouped[["desa", "diagnosa", "jumlah_kasus"]].sort_values(by="jumlah_kasus", ascending=False), use_container_width=True, hide_index=True)

    with col_grafik:
        st.markdown("#### 📊 Top Desa Terdampak")
        top_desa_df = df_grouped.groupby("desa")["jumlah_kasus"].sum().reset_index().sort_values("jumlah_kasus", ascending=False).head(10)
        st.bar_chart(top_desa_df.set_index("desa")["jumlah_kasus"])
        st.download_button("📥 Download Data Top Desa (Excel)", convert_df_to_excel(top_desa_df), "top_desa_terdampak.xlsx")

def page_pembiayaan(df_filtered, filter_info):
    st.subheader("💳 Analisis Pembiayaan")
    if df_filtered is None or "pembiayaan" not in df_filtered.columns: return
    
    df_bayar = df_filtered["pembiayaan"].value_counts().reset_index()
    df_bayar.columns = ["Pembiayaan", "Jumlah"]
    st.bar_chart(df_bayar.set_index("Pembiayaan"))
    st.download_button("📥 Download Data Pembiayaan (Excel)", convert_df_to_excel(df_bayar), "pembiayaan.xlsx")

# ================== HELPER: DETEKSI & TAMBAH HARI LIBUR NASIONAL ==================
def _get_hari_libur_indonesia(tahun_list):
    """
    Menghasilkan DataFrame hari libur nasional Indonesia yang relevan
    untuk dimasukkan sebagai regressor ke Prophet.
    Hari libur diketahui mempengaruhi pola kunjungan puskesmas secara signifikan.
    """
    libur = []
    for tahun in tahun_list:
        libur += [
            {"holiday": "tahun_baru",        "ds": f"{tahun}-01-01", "lower_window": 0, "upper_window": 1},
            {"holiday": "isra_miraj",         "ds": f"{tahun}-01-27", "lower_window": 0, "upper_window": 0},
            {"holiday": "imlek",              "ds": f"{tahun}-01-29", "lower_window": 0, "upper_window": 0},
            {"holiday": "hari_raya_nyepi",    "ds": f"{tahun}-03-20", "lower_window": 0, "upper_window": 0},
            {"holiday": "wafat_yesus",        "ds": f"{tahun}-04-18", "lower_window": 0, "upper_window": 0},
            {"holiday": "hari_buruh",         "ds": f"{tahun}-05-01", "lower_window": 0, "upper_window": 0},
            {"holiday": "kenaikan_yesus",     "ds": f"{tahun}-05-29", "lower_window": 0, "upper_window": 0},
            {"holiday": "hari_lahir_pancasila","ds": f"{tahun}-06-01", "lower_window": 0, "upper_window": 0},
            {"holiday": "idul_adha",          "ds": f"{tahun}-06-07", "lower_window": -1,"upper_window": 1},
            {"holiday": "tahun_baru_islam",   "ds": f"{tahun}-06-27", "lower_window": 0, "upper_window": 0},
            {"holiday": "hut_ri",             "ds": f"{tahun}-08-17", "lower_window": 0, "upper_window": 0},
            {"holiday": "maulid_nabi",        "ds": f"{tahun}-09-05", "lower_window": 0, "upper_window": 0},
            {"holiday": "natal",              "ds": f"{tahun}-12-25", "lower_window": -1,"upper_window": 1},
            # Lebaran: biasanya libur panjang, kunjungan turun drastis
            {"holiday": "idul_fitri",         "ds": f"{tahun}-04-10", "lower_window": -3,"upper_window": 3},
        ]
    df_libur = pd.DataFrame(libur)
    df_libur["ds"] = pd.to_datetime(df_libur["ds"])
    return df_libur


def _deteksi_outlier_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Mendeteksi outlier menggunakan metode IQR dan menggantinya dengan
    nilai median rolling 4 minggu. Outlier ekstrem (lonjakan/drop tiba-tiba)
    bisa membuat model Prophet 'bingung' dan tidak akurat.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    median_rolling = series.rolling(window=4, min_periods=1, center=True).median()
    cleaned = series.copy()
    mask = (series < lower) | (series > upper)
    cleaned[mask] = median_rolling[mask]
    return cleaned.clip(lower=0)


def _bangun_model_prophet(df_prophet: pd.DataFrame, n_minggu: int,
                           use_monthly: bool, use_outlier_cap: bool,
                           use_libur: bool, changepoint_scale: float,
                           seasonality_scale: float) -> tuple:
    """
    Membangun dan melatih model Prophet dengan semua peningkatan kualitas:
    1. Deteksi & koreksi outlier (IQR)
    2. Transformasi log untuk stabilisasi variansi
    3. Seasonality bulanan kustom
    4. Hari libur nasional Indonesia
    5. Tuning changepoint & seasonality prior scale
    6. Floor & cap (logistic growth jika variance tinggi)
    Mengembalikan (model, forecast, df_train_final)
    """
    df = df_prophet.copy()

    # ── 1. Isi minggu yang kosong (zero-fill) ──────────────────────────────
    full_range = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq="W-MON")
    df = df.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    df.columns = ["ds", "y"]

    # ── 2. Deteksi & koreksi outlier ──────────────────────────────────────
    if use_outlier_cap:
        df["y"] = _deteksi_outlier_iqr(df["y"])

    # ── 3. Tentukan floor & cap ────────────────────────────────────────────
    floor_val = 0
    cap_val   = max(df["y"].max() * 1.5, df["y"].mean() * 3)
    df["floor"] = floor_val
    df["cap"]   = cap_val

    # Cek apakah lebih cocok logistic (variance tinggi) atau linear
    cv = df["y"].std() / (df["y"].mean() + 1e-9)
    use_logistic = cv > 0.6

    growth = "logistic" if use_logistic else "linear"
    if growth == "linear":
        df = df.drop(columns=["floor", "cap"])

    # ── 4. Siapkan hari libur ─────────────────────────────────────────────
    tahun_list = list(range(df["ds"].dt.year.min(), df["ds"].dt.year.max() + 2))
    df_libur   = _get_hari_libur_indonesia(tahun_list) if use_libur else None

    # ── 5. Bangun model Prophet ───────────────────────────────────────────
    model_kwargs = dict(
        growth=growth,
        changepoint_prior_scale=changepoint_scale,   # fleksibilitas tren
        seasonality_prior_scale=seasonality_scale,   # kekuatan seasonality
        seasonality_mode="multiplicative",            # lebih akurat untuk data count
        yearly_seasonality=True if len(df) >= 52 else False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
    )
    if df_libur is not None:
        model_kwargs["holidays"] = df_libur

    model = Prophet(**model_kwargs)

    # ── 6. Tambah seasonality bulanan kustom ──────────────────────────────
    if use_monthly and len(df) >= 24:
        model.add_seasonality(
            name="monthly",
            period=30.5,
            fourier_order=5,   # lebih ekspresif dari default
            mode="multiplicative",
        )

    # ── 7. Fit ────────────────────────────────────────────────────────────
    model.fit(df)

    # ── 8. Prediksi ───────────────────────────────────────────────────────
    future = model.make_future_dataframe(periods=n_minggu, freq="W-MON")
    if growth == "logistic":
        future["floor"] = floor_val
        future["cap"]   = cap_val

    forecast = model.predict(future)
    forecast["yhat"]       = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    return model, forecast, df


# ================== HALAMAN ML (PROPHET) — VERSI AKURASI TINGGI ==================
def page_ml(df_filtered, filter_info):
    st.subheader("📈 Prediksi & Peramalan Tren (Prophet — Akurasi Tinggi)")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong.")
        return

    if "tanggal_kunjungan" not in df_filtered.columns:
        st.error("❌ Kolom 'tanggal_kunjungan' tidak ditemukan.")
        return

    df_ml = df_filtered.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_ml["tanggal_kunjungan"]):
        df_ml["tanggal_kunjungan"] = pd.to_datetime(df_ml["tanggal_kunjungan"], errors="coerce")
    df_ml = df_ml.dropna(subset=["tanggal_kunjungan"])

    if len(df_ml) == 0:
        st.warning("⚠️ Data tanggal tidak valid.")
        return

    max_date = df_ml["tanggal_kunjungan"].max().date()

    # ── Panel Pengaturan ──────────────────────────────────────────────────
    with st.container():
        col_set1, col_set2, col_set3 = st.columns([1, 1, 1])
        with col_set1:
            fokus = st.radio("Analisis:", ["Diagnosa Penyakit", "Poli / Unit"], horizontal=True, key="ml_fokus")
            kolom_fokus = "diagnosa" if fokus == "Diagnosa Penyakit" else "poli"
        with col_set2:
            if kolom_fokus not in df_ml.columns: return
            top_items = df_ml[kolom_fokus].value_counts().head(30)
            pilihan_item = st.selectbox(f"Pilih {fokus}:", options=top_items.index.tolist(), key="ml_item")
        with col_set3:
            target_date = st.date_input(
                "Prediksi Sampai Tanggal:",
                value=max_date + pd.Timedelta(days=30),
                min_value=max_date + pd.Timedelta(days=1),
                max_value=max_date + pd.Timedelta(days=365),
                key="ml_target_date"
            )

    # ── Panel Tuning Model ────────────────────────────────────────────────
    with st.expander("⚙️ Pengaturan Akurasi Model (Opsional — Default sudah optimal)", expanded=False):
        st.markdown("Pengaturan ini mempengaruhi kualitas model secara signifikan. Biarkan default jika ragu.")
        tc1, tc2 = st.columns(2)
        with tc1:
            use_outlier_cap  = st.toggle("🧹 Koreksi Outlier Otomatis (IQR)", value=True,
                help="Mendeteksi & mengoreksi lonjakan/drop ekstrem yang bisa merusak model")
            use_libur        = st.toggle("🗓️ Gunakan Hari Libur Nasional Indonesia", value=True,
                help="Kunjungan puskesmas dipengaruhi hari libur. Aktifkan untuk akurasi lebih baik.")
            use_monthly      = st.toggle("📅 Tambah Pola Musiman Bulanan", value=True,
                help="Menangkap pola berulang setiap bulan (misalnya puncak awal/akhir bulan)")
        with tc2:
            changepoint_scale = st.select_slider(
                "🔧 Fleksibilitas Tren (Changepoint Scale)",
                options=[0.01, 0.05, 0.1, 0.3, 0.5],
                value=0.1,
                help="Nilai kecil = tren lebih smooth. Nilai besar = tren mengikuti fluktuasi. Default 0.1 biasanya optimal."
            )
            seasonality_scale = st.select_slider(
                "🌊 Kekuatan Musiman (Seasonality Scale)",
                options=[1.0, 5.0, 10.0, 20.0],
                value=10.0,
                help="Nilai lebih besar = pola musiman lebih dominan. Cocok untuk data puskesmas."
            )

    # ── Persiapan Data ────────────────────────────────────────────────────
    df_item = df_ml[df_ml[kolom_fokus] == pilihan_item].copy()

    if len(df_item) < 10:
        st.error("❌ Data terlalu sedikit (< 10 pasien) untuk dipelajari oleh model.")
        return

    weekly = (
        df_item
        .groupby(pd.Grouper(key="tanggal_kunjungan", freq="W-MON"))
        .size()
        .reset_index(name="jumlah")
    )
    df_prophet_raw = weekly.rename(columns={"tanggal_kunjungan": "ds", "jumlah": "y"})

    last_date    = df_prophet_raw["ds"].max().date()
    target_dt    = pd.to_datetime(target_date).date()
    delta_days   = (target_dt - last_date).days
    periods_ahead = max(1, delta_days // 7)

    # ── Tampilkan Data Rekap ──────────────────────────────────────────────
    with st.expander("👁️ Lihat Data Rekap yang Dipelajari Model", expanded=False):
        st.dataframe(df_prophet_raw.rename(columns={"ds": "Periode (Minggu)", "y": "Jumlah Pasien"}),
                     use_container_width=True)

    # ── Training & Forecasting ────────────────────────────────────────────
    with st.spinner("🔄 Model sedang dilatih dengan semua peningkatan akurasi..."):
        try:
            model, forecast, df_prophet_clean = _bangun_model_prophet(
                df_prophet=df_prophet_raw,
                n_minggu=periods_ahead,
                use_monthly=use_monthly,
                use_outlier_cap=use_outlier_cap,
                use_libur=use_libur,
                changepoint_scale=changepoint_scale,
                seasonality_scale=seasonality_scale,
            )
        except Exception as e:
            st.error(f"❌ Gagal melatih model: {e}")
            return

    # ── Evaluasi Cepat (In-sample MAPE) ──────────────────────────────────
    forecast_insample = forecast[forecast["ds"].isin(df_prophet_clean["ds"])].copy()
    df_eval_q = pd.merge(
        df_prophet_clean[["ds", "y"]],
        forecast_insample[["ds", "yhat"]],
        on="ds", how="inner"
    )
    df_eval_q = df_eval_q[df_eval_q["y"] > 0]
    if len(df_eval_q) > 0:
        mape_insample = float(np.mean(np.abs((df_eval_q["y"] - df_eval_q["yhat"]) / df_eval_q["y"])) * 100)
        r2_insample_ss_res = float(np.sum((df_eval_q["y"] - df_eval_q["yhat"]) ** 2))
        r2_insample_ss_tot = float(np.sum((df_eval_q["y"] - df_eval_q["y"].mean()) ** 2))
        r2_insample = 1 - r2_insample_ss_res / r2_insample_ss_tot if r2_insample_ss_tot != 0 else float("nan")

        if mape_insample <= 10 and r2_insample >= 0.85:
            badge_color, badge_text = "#16a34a", f"🟢 Sangat Baik (MAPE {mape_insample:.1f}%, R² {r2_insample:.3f})"
        elif mape_insample <= 20 and r2_insample >= 0.70:
            badge_color, badge_text = "#ca8a04", f"🟡 Cukup Baik (MAPE {mape_insample:.1f}%, R² {r2_insample:.3f})"
        else:
            badge_color, badge_text = "#dc2626", f"🔴 Perlu Data Lebih Banyak (MAPE {mape_insample:.1f}%)"

        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:0.6rem;padding:0.35rem 1rem;
             border-radius:999px;background:{badge_color}22;border:1.5px solid {badge_color};
             color:{badge_color};font-weight:700;font-size:0.95rem;margin-bottom:0.5rem;">
            Kualitas Fit Model: {badge_text}
        </div>
        """, unsafe_allow_html=True)

    # ── Grafik Utama ──────────────────────────────────────────────────────
    st.markdown(f"### 📈 Grafik Peramalan: **{pilihan_item}**")

    forecast_future = forecast[forecast["ds"] > pd.to_datetime(last_date)].copy()
    forecast_hist   = forecast[forecast["ds"] <= pd.to_datetime(last_date)].copy()

    fig = go.Figure()

    # Area kepercayaan prediksi masa depan
    fig.add_trace(go.Scatter(
        x=forecast_future["ds"].tolist() + forecast_future["ds"].tolist()[::-1],
        y=forecast_future["yhat_upper"].tolist() + forecast_future["yhat_lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Rentang Kepercayaan 80%", hoverinfo="skip",
    ))

    # Fit model pada data historis (garis merah halus)
    fig.add_trace(go.Scatter(
        x=forecast_hist["ds"], y=forecast_hist["yhat"],
        mode="lines", name="Fit Model (Historis)",
        line=dict(color="rgba(239,68,68,0.5)", width=1.5, dash="dot"),
    ))

    # Data aktual (setelah outlier dikoreksi)
    fig.add_trace(go.Scatter(
        x=df_prophet_clean["ds"], y=df_prophet_clean["y"],
        mode="lines+markers", name="Data Aktual",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=5),
    ))

    # Prediksi masa depan
    fig.add_trace(go.Scatter(
        x=forecast_future["ds"], y=forecast_future["yhat"],
        mode="lines+markers", name="Prediksi Masa Depan",
        line=dict(color="#ef4444", width=3, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
    ))

    fig.update_layout(
        xaxis_title="Periode Waktu", yaxis_title="Jumlah Kunjungan/Pasien",
        hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Grafik Komponen Seasonality ───────────────────────────────────────
    with st.expander("🔍 Lihat Komponen Pola Model (Tren, Musiman, Libur)", expanded=False):
        st.markdown("Grafik di bawah menunjukkan kontribusi masing-masing komponen yang dipelajari model.")
        try:
            fig_comp = model.plot_components(forecast)
            st.pyplot(fig_comp, use_container_width=True)
        except Exception:
            st.info("Grafik komponen tidak tersedia untuk konfigurasi model ini.")

    # ── Download Prediksi ─────────────────────────────────────────────────
    df_download_pred = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    df_download_pred.columns = ["Tanggal", "Prediksi_Jumlah", "Batas_Bawah", "Batas_Atas"]
    st.download_button(
        label="📥 Download Data Prediksi (Excel)",
        data=convert_df_to_excel(df_download_pred),
        file_name=f"prediksi_{kolom_fokus}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Kesimpulan Estimasi ───────────────────────────────────────────────
    st.markdown(f"### 📢 Kesimpulan Estimasi Hingga {pd.to_datetime(target_date).strftime('%d %B %Y')}")
    if not forecast_future.empty:
        total_estimasi       = int(round(forecast_future["yhat"].clip(lower=0).sum()))
        target_week          = forecast_future.iloc[-1]
        tgl_target_akhir     = target_week["ds"].strftime("%d %B %Y")
        est_kunjungan_akhir  = max(0, int(round(target_week["yhat"])))
        batas_bawah_akhir    = max(0, int(round(target_week["yhat_lower"])))
        batas_atas_akhir     = max(0, int(round(target_week["yhat_upper"])))

        col_alert, col_metric1, col_metric2 = st.columns([2, 1, 1])
        with col_alert:
            st.markdown(f"""
                <div style="padding:1rem;border-radius:0.5rem;background-color:rgba(59,130,246,0.1);
                     border:1px solid rgba(59,130,246,0.3);margin-bottom:1rem;">
                    <div class="highlight-estimasi">
                        📆 Selama periode ke depan hingga <b>{tgl_target_akhir}</b>, model memperkirakan
                        akan ada <b>total akumulasi {total_estimasi} kunjungan/kasus</b> untuk
                        <b>{pilihan_item}</b>.<br><br>
                        Khusus pada pekan terakhir tersebut, diprediksi terdapat
                        <b>{est_kunjungan_akhir} kunjungan</b> baru
                        (rentang: {batas_bawah_akhir}–{batas_atas_akhir}).
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_metric1:
            st.metric("Total Akumulasi Pasien", f"{total_estimasi} Pasien")
        with col_metric2:
            st.metric("Estimasi Pekan Terakhir", f"{est_kunjungan_akhir} Pasien",
                      help=f"Batas Bawah: {batas_bawah_akhir} | Batas Atas: {batas_atas_akhir}")
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

# ================== HALAMAN Agent AI ==================
def page_ai_assistant(df_filtered, filter_info, is_genai_configured):
    st.subheader("🤖 Agent AI Cerdas")
    
    if df_filtered is None or df_filtered.empty: 
        st.warning("⚠️ Data belum ada atau kosong. Silakan upload dan filter data terlebih dahulu di panel kiri.")
        return
    if not is_genai_configured: 
        st.error("❌ API Key belum diset. Agent AI tidak dapat digunakan.")
        return

    total_data = len(df_filtered)
    
    top_diagnosa = "-"
    if "diagnosa" in df_filtered.columns:
        top_diagnosa_series = df_filtered["diagnosa"].value_counts().head(5)
        top_diagnosa = ", ".join([f"{k} ({v} kasus)" for k, v in top_diagnosa_series.items()])
        
    top_poli = "-"
    if "poli" in df_filtered.columns:
        top_poli_series = df_filtered["poli"].value_counts().head(3)
        top_poli = ", ".join([f"{k} ({v} kunjungan)" for k, v in top_poli_series.items()])
        
    gender = df_filtered['jenis_kelamin'].mode()[0] if 'jenis_kelamin' in df_filtered.columns and not df_filtered['jenis_kelamin'].dropna().empty else "-"
    umur = df_filtered['kelompok_umur'].mode()[0] if 'kelompok_umur' in df_filtered.columns and not df_filtered['kelompok_umur'].dropna().empty else "-"
    
    desa = "-"
    if "desa" in df_filtered.columns:
        desa_series = df_filtered["desa"].value_counts().head(3)
        desa = ", ".join([f"{k} ({v} pasien)" for k, v in desa_series.items()])

    context_summary = f"""[STATISTIK DATA PASIEN SAAT INI]
- Total Pasien/Kunjungan: {total_data}
- 5 Penyakit Terbanyak: {top_diagnosa}
- 3 Poli Terpadat: {top_poli}
- Demografi Mayoritas: Jenis Kelamin {gender}, Kelompok Usia {umur}
- 3 Desa Asal Pasien Terbanyak: {desa}"""

    with st.expander("👁️ Lihat Konteks Data yang Akan Dikirim ke AI", expanded=False):
        st.markdown("Di balik layar, aplikasi akan menyisipkan ringkasan teks ini kepada AI agar jawaban yang diberikan akurat dan sesuai dengan kondisi Puskesmas berdasarkan filter Anda saat ini:")
        st.code(context_summary, language="markdown")

    st.info('💡 **Tips:** Coba tanyakan: *"Berdasarkan data penyakit terbanyak saat ini, program promkes desa apa yang paling mendesak?"* atau *"Apa obat yang perlu saya siapkan lebih banyak bulan ini?"*')

    user_q = st.text_area("Tanyakan strategi/analisis kesehatan:", placeholder="Ketik pertanyaan Anda di sini...")
    
    if st.button("Kirim Pertanyaan"):
        if not user_q.strip():
            st.warning("Pertanyaan tidak boleh kosong.")
            return

        prompt = f"""Anda adalah Analis Data dan Ahli Kesehatan Masyarakat profesional di UPT Puskesmas Purwosari (Bojonegoro). 
        Berikut adalah ringkasan data pasien secara real-time berdasarkan filter yang sedang aktif di aplikasi:
        
        {context_summary}
        
        Tugas Anda: Jawablah pertanyaan pengguna di bawah ini secara spesifik, berbasis pada data di atas, praktis, dan terstruktur. Jangan berhalusinasi, gunakan angka-angka di atas sebagai landasan argumen Anda.
        
        Pertanyaan Pengguna: {user_q}
        """
        
        with st.spinner("🤖 AI sedang menganalisis data dan merumuskan jawaban..."):
            try:
                model = genai.GenerativeModel('gemini-2.5-flash') 
                response = model.generate_content(prompt)
                
                st.markdown("### 📊 Analisis AI:")
                st.markdown(f"""
                <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: var(--secondary-background-color); border: 1px solid rgba(148, 163, 184, 0.4);">
                    {response.text}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    st.warning("⏳ **Sistem AI sedang sibuk (Limit Penggunaan).** Silakan tunggu sekitar 1 menit, lalu klik 'Kirim' lagi.")
                else:
                    st.error(f"❌ Gagal terhubung ke server AI. Detail: {error_msg}")

# ================== HALAMAN CETAK LAPORAN PDF ==================
def page_cetak_laporan(df_filtered, filter_info):
    st.subheader("🖨️ Cetak Laporan PDF (Lengkap dengan Grafik)")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong. Silakan upload dan atur filter data terlebih dahulu.")
        return

    st.info("💡 **Catatan:** Proses pembuatan PDF dengan grafik akan memakan waktu beberapa detik karena sistem harus merender visualisasi data terlebih dahulu.")

    if st.button("📄 Buat Dokumen PDF"):
        with st.spinner("Menyusun laporan dan merender grafik..."):
            try:
                total_kunjungan = len(df_filtered)
                
                top_diagnosa = []
                if "diagnosa" in df_filtered.columns:
                    top_diagnosa = df_filtered["diagnosa"].value_counts().head(5).items()
                    
                top_poli = []
                if "poli" in df_filtered.columns:
                    top_poli = df_filtered["poli"].value_counts().head(5).items()

                pdf = FPDF()
                pdf.add_page()
                
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 8, "LAPORAN RINGKASAN KUNJUNGAN PASIEN", ln=True, align='C')
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 8, "UPT PUSKESMAS PURWOSARI - KAB. BOJONEGORO", ln=True, align='C')
                pdf.set_font("Arial", '', 10)
                tanggal_cetak = datetime.now().strftime("%d %B %Y %H:%M")
                pdf.cell(0, 6, f"Waktu Cetak: {tanggal_cetak}", ln=True, align='C')
                pdf.line(10, 35, 200, 35)
                pdf.ln(10)

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "1. Parameter Filter Aktif", ln=True)
                pdf.set_font("Arial", '', 11)
                if filter_info and any(filter_info.values()):
                    for k, v in filter_info.items():
                        if v:
                            label_bersih = k.replace("_", " ").title()
                            pdf.cell(0, 6, f"- {label_bersih}: {', '.join(map(str, v))}", ln=True)
                else:
                    pdf.cell(0, 6, "- Menampilkan Semua Data (Tidak ada filter)", ln=True)
                pdf.ln(5)

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "2. Ringkasan Kunjungan", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.cell(0, 6, f"Total Kunjungan Pasien : {total_kunjungan} kunjungan", ln=True)
                if "no_rm" in df_filtered.columns:
                    pdf.cell(0, 6, f"Total Pasien Unik (RM) : {df_filtered['no_rm'].nunique()} pasien", ln=True)
                pdf.ln(5)

                if top_diagnosa:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "3. Top 5 Diagnosa Penyakit Terbanyak", ln=True)
                    pdf.set_font("Arial", '', 11)
                    for i, (penyakit, jumlah) in enumerate(top_diagnosa, 1):
                        pdf.cell(0, 6, f"{i}. {penyakit} ({jumlah} kasus)", ln=True)
                    pdf.ln(5)

                if "jenis_kelamin" in df_filtered.columns:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "4. Demografi Jenis Kelamin", ln=True)
                    pdf.set_font("Arial", '', 11)
                    jk_counts = df_filtered["jenis_kelamin"].value_counts()
                    for jk, jml in jk_counts.items():
                        pdf.cell(0, 6, f"- {jk}: {jml} pasien", ln=True)

                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "LAMPIRAN VISUALISASI DATA", ln=True, align='C')
                pdf.line(10, 20, 200, 20)
                pdf.ln(5)

                temp_images = []

                if "diagnosa" in df_filtered.columns:
                    top10_df = df_filtered["diagnosa"].value_counts().head(10).reset_index()
                    top10_df.columns = ['Diagnosa', 'Jumlah']
                    fig_diag = px.bar(top10_df, x='Diagnosa', y='Jumlah', title="Top 10 Diagnosa Penyakit", text_auto=True)
                    
                    fd, path_diag = tempfile.mkstemp(suffix=".png")
                    fig_diag.write_image(path_diag, engine="kaleido", width=800, height=500)
                    temp_images.append(path_diag)

                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "A. Grafik Distribusi Penyakit", ln=True)
                    pdf.image(path_diag, x=15, w=180)
                    pdf.ln(5)

                if "tanggal_kunjungan" in df_filtered.columns:
                    trend = df_filtered.groupby(["tahun", "bulan", "nama_bulan"]).size().reset_index(name="count").sort_values(["tahun", "bulan"])
                    if not trend.empty:
                        trend["Periode"] = trend["nama_bulan"].astype(str) + " " + trend["tahun"].astype(str)
                        fig_trend = px.line(trend, x="Periode", y="count", title="Tren Kunjungan Pasien Berdasarkan Bulan", markers=True)
                        
                        fd, path_trend = tempfile.mkstemp(suffix=".png")
                        fig_trend.write_image(path_trend, engine="kaleido", width=800, height=400)
                        temp_images.append(path_trend)

                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 8, "B. Tren Kunjungan Bulanan", ln=True)
                        pdf.image(path_trend, x=15, w=180)
                        pdf.ln(5)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    pdf.output(tmp_pdf.name)
                    with open(tmp_pdf.name, "rb") as f:
                        pdf_bytes = f.read()

                for img_path in temp_images:
                    if os.path.exists(img_path):
                        os.remove(img_path)

                st.success("✅ Laporan PDF beserta grafik berhasil dibuat!")
                
                st.download_button(
                    label="📥 Download Laporan PDF (Teks & Grafik)",
                    data=pdf_bytes,
                    file_name=f"Laporan_Komprehensif_Puskesmas_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses grafik. Pastikan Anda sudah menambahkan 'kaleido' di requirements.txt. Detail error: {e}")

# ================== HALAMAN EVALUASI AKURASI MODEL ==================

def _hitung_metrik(y_actual, y_pred):
    """Menghitung MAE, RMSE, MAPE, dan R² antara nilai aktual vs prediksi."""
    y_actual = np.array(y_actual, dtype=float)
    y_pred   = np.array(y_pred,   dtype=float)

    mae  = float(np.mean(np.abs(y_actual - y_pred)))
    rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))

    mask = y_actual != 0
    mape = float(np.mean(np.abs((y_actual[mask] - y_pred[mask]) / y_actual[mask])) * 100) if mask.any() else float("nan")

    ss_res = float(np.sum((y_actual - y_pred) ** 2))
    ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R²": r2}


def _interpretasi_akurasi(mape, r2):
    """Mengembalikan (label_kualitas, warna_badge) berdasarkan MAPE & R²."""
    if mape <= 10 and r2 >= 0.85:
        return "🟢 Sangat Baik", "#16a34a"
    elif mape <= 20 and r2 >= 0.70:
        return "🟡 Cukup Baik", "#ca8a04"
    elif mape <= 30:
        return "🟠 Perlu Perhatian", "#ea580c"
    else:
        return "🔴 Kurang Akurat", "#dc2626"


def page_evaluasi_akurasi(df_filtered, filter_info):
    """
    Halaman evaluasi akurasi model Prophet menggunakan teknik
    Train/Test Split (Holdout Validation).
    - 80% data lama → training
    - 20% data terbaru → testing (dibandingkan dengan prediksi)
    - Metrik: MAE, RMSE, MAPE, R²
    """
    st.subheader("📐 Evaluasi Akurasi Model Prediksi (Prophet)")

    with st.expander("ℹ️ Apa itu Evaluasi Akurasi Model & Cara Kerjanya?", expanded=False):
        st.markdown("""
        **Mengapa perlu evaluasi?**
        Model Prophet menghasilkan prediksi, tetapi kita perlu tahu *seberapa jauh* prediksi itu
        meleset dari data nyata. Itulah gunanya evaluasi akurasi.

        **Metode: Holdout Validation (Train/Test Split)**
        | Tahap | Penjelasan |
        |---|---|
        | **Training Set (80%)** | Data lama yang digunakan AI untuk "belajar" pola |
        | **Test Set (20%)** | Data terbaru yang *disembunyikan* dari AI, lalu dipakai untuk menguji |

        **Metrik yang digunakan:**
        | Metrik | Arti | Target Ideal |
        |---|---|---|
        | **MAE** | Rata-rata selisih absolut prediksi vs aktual (satuan: pasien) | Sekecil mungkin |
        | **RMSE** | Seperti MAE, tapi lebih sensitif terhadap kesalahan besar | Sekecil mungkin |
        | **MAPE (%)** | Rata-rata persentase kesalahan prediksi | < 10% = Sangat Baik |
        | **R²** | Seberapa baik model menjelaskan variasi data (0–1) | > 0.85 = Sangat Baik |

        > 💡 **Contoh:** MAPE 8% berarti rata-rata prediksi meleset 8% dari angka aktual.
        > Untuk konteks puskesmas, MAPE di bawah 20% sudah sangat layak digunakan sebagai
        > dasar perencanaan logistik dan anggaran.
        """)

    if df_filtered is None or df_filtered.empty:
        st.warning("⚠️ Data belum tersedia. Silakan upload file terlebih dahulu.")
        return

    if "tanggal_kunjungan" not in df_filtered.columns:
        st.error("❌ Kolom 'tanggal_kunjungan' tidak ditemukan.")
        return

    df_ml = df_filtered.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_ml["tanggal_kunjungan"]):
        df_ml["tanggal_kunjungan"] = pd.to_datetime(df_ml["tanggal_kunjungan"], errors="coerce")
    df_ml = df_ml.dropna(subset=["tanggal_kunjungan"])

    if len(df_ml) < 30:
        st.warning("⚠️ Data terlalu sedikit (minimal 30 baris) untuk evaluasi yang bermakna.")
        return

    st.markdown("---")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        fokus_eval = st.radio("Analisis berdasarkan:", ["Diagnosa Penyakit", "Poli / Unit"], horizontal=True, key="eval_fokus")
        kolom_fokus_eval = "diagnosa" if fokus_eval == "Diagnosa Penyakit" else "poli"
    with col_b:
        if kolom_fokus_eval not in df_ml.columns:
            st.error(f"❌ Kolom '{kolom_fokus_eval}' tidak ditemukan dalam data.")
            return
        top_items_eval = df_ml[kolom_fokus_eval].value_counts().head(20)
        pilihan_item_eval = st.selectbox(f"Pilih {fokus_eval}:", options=top_items_eval.index.tolist(), key="eval_item")
    with col_c:
        test_pct = st.slider("Proporsi Data Uji (%)", min_value=10, max_value=40, value=20, step=5, key="eval_split",
                             help="Persentase data terbaru yang digunakan sebagai data uji.")

    df_item_eval = df_ml[df_ml[kolom_fokus_eval] == pilihan_item_eval].copy()

    if len(df_item_eval) < 20:
        st.error(f"❌ Data untuk '{pilihan_item_eval}' terlalu sedikit (< 20 baris). Coba pilih diagnosa/poli lain.")
        return

    weekly_eval = (
        df_item_eval
        .groupby(pd.Grouper(key="tanggal_kunjungan", freq="W-MON"))
        .size()
        .reset_index(name="jumlah")
    )
    df_prophet_eval = weekly_eval.rename(columns={"tanggal_kunjungan": "ds", "jumlah": "y"})
    df_prophet_eval = df_prophet_eval[df_prophet_eval["y"] > 0].reset_index(drop=True)

    if len(df_prophet_eval) < 15:
        st.error(f"❌ Data mingguan untuk '{pilihan_item_eval}' terlalu sedikit ({len(df_prophet_eval)} minggu). Minimal 15 minggu.")
        return

    n_total  = len(df_prophet_eval)
    n_test   = max(4, int(n_total * test_pct / 100))
    n_train  = n_total - n_test

    df_train_eval = df_prophet_eval.iloc[:n_train].copy()
    df_test_eval  = df_prophet_eval.iloc[n_train:].copy()

    col_i1, col_i2, col_i3 = st.columns(3)
    col_i1.metric("Total Minggu Data", f"{n_total} minggu")
    col_i2.metric("Training Set", f"{n_train} minggu", delta=f"{100 - test_pct}% dari data")
    col_i3.metric("Test Set", f"{n_test} minggu", delta=f"{test_pct}% dari data")

    st.markdown(f"""
    > **Garis batas split:** Training hingga **{df_train_eval['ds'].max().strftime('%d %b %Y')}**  
    > Testing dari **{df_test_eval['ds'].min().strftime('%d %b %Y')}** sampai **{df_test_eval['ds'].max().strftime('%d %b %Y')}**
    """)

    with st.spinner(f"🔄 Model sedang belajar dari {n_train} minggu data training..."):
        try:
            model_eval, forecast_eval_full, df_train_clean = _bangun_model_prophet(
                df_prophet=df_train_eval,
                n_minggu=n_test,
                use_monthly=True,
                use_outlier_cap=True,
                use_libur=True,
                changepoint_scale=0.1,
                seasonality_scale=10.0,
            )
        except Exception as e:
            st.error(f"❌ Gagal melatih model evaluasi: {e}")
            return
        forecast_test_eval = forecast_eval_full[forecast_eval_full["ds"].isin(df_test_eval["ds"])].copy()

    if forecast_test_eval.empty:
        st.error("❌ Prediksi gagal dihasilkan. Coba kurangi proporsi data uji.")
        return

    df_eval = pd.merge(
        df_test_eval[["ds", "y"]].rename(columns={"y": "Aktual"}),
        forecast_test_eval[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={"yhat": "Prediksi", "yhat_lower": "Batas_Bawah", "yhat_upper": "Batas_Atas"}
        ),
        on="ds", how="inner"
    )
    df_eval["Prediksi"]    = df_eval["Prediksi"].clip(lower=0).round(1)
    df_eval["Batas_Bawah"] = df_eval["Batas_Bawah"].clip(lower=0).round(1)
    df_eval["Batas_Atas"]  = df_eval["Batas_Atas"].clip(lower=0).round(1)
    df_eval["Selisih"]     = (df_eval["Prediksi"] - df_eval["Aktual"]).round(1)
    df_eval["Error (%)"]   = np.where(
        df_eval["Aktual"] != 0,
        ((df_eval["Prediksi"] - df_eval["Aktual"]).abs() / df_eval["Aktual"] * 100).round(1),
        np.nan
    )

    metrik = _hitung_metrik(df_eval["Aktual"].values, df_eval["Prediksi"].values)
    label_kualitas, warna_badge = _interpretasi_akurasi(metrik["MAPE (%)"], metrik["R²"])

    st.markdown("---")
    st.markdown(f"### 📊 Hasil Evaluasi Model — **{pilihan_item_eval}**")

    st.markdown(f"""
    <div style="display:inline-block; padding: 0.4rem 1.2rem; border-radius: 999px;
         background-color: {warna_badge}22; border: 2px solid {warna_badge};
         color: {warna_badge}; font-weight: 700; font-size: 1.05rem; margin-bottom: 1rem;">
        Kualitas Model: {label_kualitas}
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",  f"{metrik['MAE']:.2f} pasien",
              help="Mean Absolute Error: rata-rata selisih absolut prediksi vs aktual")
    m2.metric("RMSE", f"{metrik['RMSE']:.2f} pasien",
              help="Root Mean Square Error: lebih sensitif terhadap kesalahan besar")
    m3.metric("MAPE", f"{metrik['MAPE (%)']:.1f}%",
              delta="< 10% = Sangat Baik" if metrik["MAPE (%)"] <= 10 else "Target < 10%",
              delta_color="normal" if metrik["MAPE (%)"] <= 10 else "inverse",
              help="Mean Absolute Percentage Error: rata-rata persentase kesalahan")
    m4.metric("R²",   f"{metrik['R²']:.3f}",
              delta="≥ 0.85 = Sangat Baik" if metrik["R²"] >= 0.85 else "Target ≥ 0.85",
              delta_color="normal" if metrik["R²"] >= 0.85 else "inverse",
              help="Koefisien determinasi: seberapa baik model menjelaskan variasi data")

    st.markdown("---")
    st.markdown("#### 📈 Grafik Aktual vs Prediksi (Periode Test)")

    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(
        x=df_eval["ds"].tolist() + df_eval["ds"].tolist()[::-1],
        y=df_eval["Batas_Atas"].tolist() + df_eval["Batas_Bawah"].tolist()[::-1],
        fill="toself", fillcolor="rgba(239, 68, 68, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Rentang Kepercayaan 80%", hoverinfo="skip",
    ))
    fig_eval.add_trace(go.Scatter(
        x=df_eval["ds"], y=df_eval["Aktual"],
        mode="lines+markers", name="Data Aktual (Nyata)",
        line=dict(color="#2563eb", width=2.5), marker=dict(size=7),
    ))
    fig_eval.add_trace(go.Scatter(
        x=df_eval["ds"], y=df_eval["Prediksi"],
        mode="lines+markers", name="Prediksi Model",
        line=dict(color="#ef4444", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
    ))
    fig_eval.update_layout(
        xaxis_title="Periode (Minggu)", yaxis_title="Jumlah Kunjungan",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0), height=380,
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    st.markdown("#### 📉 Grafik Residual (Selisih Prediksi vs Aktual)")
    st.caption("Garis mendekati nol artinya prediksi makin akurat. Pola sistematis menunjukkan model masih bisa diperbaiki.")

    colors_bar = ["#16a34a" if v >= 0 else "#dc2626" for v in df_eval["Selisih"]]
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Bar(
        x=df_eval["ds"], y=df_eval["Selisih"],
        marker_color=colors_bar, name="Selisih (Prediksi − Aktual)",
        hovertemplate="<b>%{x}</b><br>Selisih: %{y:.1f} pasien<extra></extra>",
    ))
    fig_resid.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1.5)
    fig_resid.update_layout(
        xaxis_title="Periode (Minggu)", yaxis_title="Selisih (pasien)",
        height=280, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    with st.expander("📋 Lihat Tabel Detail Aktual vs Prediksi", expanded=False):
        df_tampil = df_eval.copy()
        df_tampil["ds"] = df_tampil["ds"].dt.strftime("%d %b %Y")
        df_tampil.columns = ["Minggu", "Aktual", "Prediksi", "Batas Bawah", "Batas Atas", "Selisih", "Error (%)"]
        st.dataframe(df_tampil, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 💡 Interpretasi & Rekomendasi")

    mape_val = metrik["MAPE (%)"]
    r2_val   = metrik["R²"]
    mae_val  = metrik["MAE"]

    if mape_val <= 10 and r2_val >= 0.85:
        rekomendasi = f"""
        ✅ **Model memiliki akurasi tinggi** (MAPE {mape_val:.1f}%, R² {r2_val:.3f}).<br><br>
        - Prediksi ini <b>layak dijadikan dasar perencanaan</b> logistik obat dan anggaran.<br>
        - Rata-rata kesalahan prediksi hanya <b>{mae_val:.1f} pasien/minggu</b> — sangat kecil.<br>
        - Model dapat digunakan langsung di halaman <b>Prediksi ML</b>.
        """
    elif mape_val <= 20 and r2_val >= 0.70:
        rekomendasi = f"""
        ⚠️ <b>Model memiliki akurasi cukup baik</b> (MAPE {mape_val:.1f}%, R² {r2_val:.3f}).<br><br>
        - Prediksi masih <b>dapat digunakan sebagai gambaran umum</b>, namun tambahkan buffer ±{mae_val:.0f} pasien/minggu.<br>
        - Pertimbangkan untuk <b>menambah data historis</b> (minimal 1 tahun) agar model lebih mengenal pola musiman.<br>
        - Coba filter data pada poli/diagnosa spesifik untuk meningkatkan akurasi.
        """
    elif mape_val <= 30:
        rekomendasi = f"""
        🔶 <b>Model perlu perbaikan</b> (MAPE {mape_val:.1f}%, R² {r2_val:.3f}).<br><br>
        - Gunakan prediksi ini <b>hanya sebagai referensi kasar</b>, bukan dasar pengambilan keputusan utama.<br>
        - <b>Saran:</b> tambahkan data historis lebih panjang, atau coba analisis pada diagnosa/poli dengan kasus lebih banyak.
        """
    else:
        rekomendasi = f"""
        ❌ <b>Model kurang akurat</b> (MAPE {mape_val:.1f}%, R² {r2_val:.3f}).<br><br>
        - <b>Jangan gunakan</b> hasil prediksi ini untuk keputusan penting.<br>
        - Kemungkinan penyebab: data sangat sedikit, pola tidak beraturan, atau banyak nilai ekstrem.<br>
        - <b>Saran:</b> pilih diagnosa/poli dengan kunjungan lebih tinggi dan data lebih konsisten.
        """

    st.markdown(f"""
    <div style="padding: 1rem 1.5rem; border-radius: 0.5rem;
         background-color: rgba(59, 130, 246, 0.07);
         border-left: 4px solid {warna_badge};">
        {rekomendasi}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    df_download_eval = df_eval.copy()
    df_download_eval["ds"] = df_download_eval["ds"].dt.strftime("%Y-%m-%d")
    df_download_eval.columns = ["Tanggal", "Aktual", "Prediksi", "Batas_Bawah", "Batas_Atas", "Selisih", "Error_Persen"]

    df_metrik_eval = pd.DataFrame([{
        "Metrik": k,
        "Nilai": round(v, 4),
        "Satuan": "pasien" if k in ["MAE", "RMSE"] else ("%" if "%" in k else "")
    } for k, v in metrik.items()])

    output_eval = io.BytesIO()
    with pd.ExcelWriter(output_eval, engine="openpyxl") as writer:
        df_download_eval.to_excel(writer, index=False, sheet_name="Aktual_vs_Prediksi")
        df_metrik_eval.to_excel(writer, index=False, sheet_name="Metrik_Akurasi")
    output_eval.seek(0)

    st.download_button(
        label="📥 Download Laporan Evaluasi (Excel)",
        data=output_eval.getvalue(),
        file_name=f"evaluasi_akurasi_{pilihan_item_eval.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ========= MAIN =========

def main():
    df_filtered, filter_info = apply_filters(None)
    
    is_genai_configured = get_gemini_client()

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigasi", [
        "Ringkasan Umum", "Analisis Kunjungan", "Analisis Penyakit", 
        "Peta Persebaran", "Analisis Pembiayaan", "Data & Unduhan", 
        "Kualitas Data", "Prediksi ML", "📐 Evaluasi Akurasi Model",
        "Agent AI", "Cetak Laporan PDF"
    ])
    
    if page == "Ringkasan Umum": page_overview(df_filtered, filter_info)
    elif page == "Analisis Kunjungan": page_kunjungan(df_filtered, filter_info)
    elif page == "Analisis Penyakit": page_penyakit(df_filtered, filter_info)
    elif page == "Peta Persebaran": page_peta_persebaran(df_filtered, filter_info)
    elif page == "Analisis Pembiayaan": page_pembiayaan(df_filtered, filter_info)
    elif page == "Data & Unduhan": page_data(df_filtered, filter_info)
    elif page == "Kualitas Data": page_quality(df_filtered)
    elif page == "Prediksi ML": page_ml(df_filtered, filter_info)
    elif page == "📐 Evaluasi Akurasi Model": page_evaluasi_akurasi(df_filtered, filter_info)
    elif page == "Agent AI": page_ai_assistant(df_filtered, filter_info, is_genai_configured)
    elif page == "Cetak Laporan PDF": page_cetak_laporan(df_filtered, filter_info)

if __name__ == "__main__":
    main()
