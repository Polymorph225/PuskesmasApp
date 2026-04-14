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
import warnings
warnings.filterwarnings("ignore")

# Library Machine Learning & AI
import google.generativeai as genai
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="Dashboard Analisis Data Puskesmas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== TAMPILAN (CSS) ==========
def inject_custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem; padding-bottom: 3rem;
            padding-left: 2rem; padding-right: 2rem;
        }
        h1 { font-weight: 700 !important; }
        div[data-testid="metric-container"] {
            padding: 0.75rem 1rem; border-radius: 0.75rem;
            background-color: var(--secondary-background-color);
            border: 1px solid var(--faded-text-color);
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 0.5rem; border: 1px solid var(--faded-text-color);
        }
        hr { margin: 0.75rem 0 1rem 0; }
        .highlight-estimasi {
            color: #1d4ed8 !important; line-height: 1.5;
            font-size: 1.05rem; font-weight: 800;
        }
        .akurasi-badge {
            display: inline-block; padding: 0.4rem 1rem;
            border-radius: 999px; font-size: 1rem; font-weight: 700; margin-top: 0.5rem;
        }
        .akurasi-hijau  { background:#dcfce7; color:#166534; border:1px solid #86efac; }
        .akurasi-kuning { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
        .akurasi-oranye { background:#ffedd5; color:#9a3412; border:1px solid #fdba74; }
        .akurasi-merah  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }
        .model-card {
            padding: 1rem; border-radius: 0.75rem; margin-bottom: 0.75rem;
            border: 2px solid transparent;
        }
        .model-card-best   { border-color: #22c55e; background: rgba(34,197,94,0.07); }
        .model-card-normal { border-color: var(--faded-text-color); background: var(--secondary-background-color); }
        .opt-banner {
            padding: 0.65rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem;
            background: rgba(59,130,246,0.08); border-left: 4px solid #3b82f6;
            font-size: 0.95rem; line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

# ========== HEADER ==========
st.title("📊 Dashboard Analisis Data Puskesmas")
st.markdown(
    """
    Dashboard ini membantu menganalisis **data kunjungan Puskesmas** berdasarkan:
    Poli / unit layanan, Diagnosa, Jenis kelamin & kelompok umur, Jenis pembiayaan, dan Wilayah pelayanan.

    ⬅️ **Mulai dengan meng-upload file data di sidebar, lalu atur filter sesuai kebutuhan.**
    """
)

# ========= GEMINI CLIENT =========
@st.cache_resource
def get_gemini_client():
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ API key Gemini belum diset. Fitur AI Agent tidak akan berfungsi.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Gagal inisialisasi Gemini API: {e}")
        return False

# ========= HELPER EXCEL =========
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

# ========= DATA CLEANING =========
TARGET_COLS = ["tanggal_kunjungan","no_rm","umur","jenis_kelamin","poli","diagnosa","pembiayaan","desa"]

def _normalize_col_name(col):
    col = str(col).strip().lower()
    return re.sub(r"[^a-z0-9]", "", col)

def _build_column_mapping(df_raw):
    mapping = {}
    alias_dict = {
        "tanggal_kunjungan": ["tanggalkunjungan","tglkunjungan","tanggal","tgl","visitdate"],
        "no_rm": ["norm","norekammedis","norekam_medis","no_rekam_medis","rekammedis"],
        "umur": ["umur","usia","age","umurth"],
        "jenis_kelamin": ["jeniskelamin","jk","sex","kelamin","llp"],
        "poli": ["poli","politujuan","unit","unitlayanan","poliklinik"],
        "diagnosa": ["diagnosa","diagnosautama","dxutama","diag","icd10","diagnosis"],
        "pembiayaan": ["pembiayaan","carabayar","penjamin","jaminan","pembayar"],
        "desa": ["desa","alamatdesa","kelurahan","desakelurahan","namadesa"],
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
    if v in ["l","lk","laki","laki-laki","male","m","1"]: return "Laki-Laki"
    if v in ["p","pr","perempuan","wanita","female","f","2"]: return "Perempuan"
    return str(val).title()

def _parse_umur(val):
    if pd.isna(val): return np.nan
    m = re.search(r"(\d+)", str(val))
    if m:
        try: return int(m.group(1))
        except: return np.nan
    try: return int(float(str(val)))
    except: return np.nan

def clean_raw_data(df_raw):
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
    for col in ["poli","diagnosa","pembiayaan","desa","no_rm"]:
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
    elif ext in ["xlsx","xls"]:
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
                df["desa"] = df[c]; break
    if "umur" in df.columns:
        df["umur"] = pd.to_numeric(df["umur"], errors="coerce")
        bins   = [0,1,5,15,25,45,60,200]
        labels = ["<1 th","1-4 th","5-14 th","15-24 th","25-44 th","45-59 th","60+ th"]
        df["kelompok_umur"] = pd.cut(df["umur"], bins=bins, labels=labels, right=False)
    if "tanggal_kunjungan" in df.columns:
        df["tahun"]      = df["tanggal_kunjungan"].dt.year
        df["bulan"]      = df["tanggal_kunjungan"].dt.month
        df["nama_bulan"] = df["tanggal_kunjungan"].dt.strftime("%b")
    for col in ["poli","jenis_kelamin","pembiayaan","diagnosa","desa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

def apply_filters(_):
    with st.sidebar:
        st.markdown("## 🏥 Data Kunjungan Puskesmas")
        uploaded_file = st.file_uploader("Upload data kunjungan (CSV / Excel)", type=["csv","xlsx","xls"])
        if uploaded_file is None:
            st.info("Silakan upload file **CSV/Excel**.")
            return None, None
        df_clean, df_raw = load_data(uploaded_file)
        df = preprocess_data(df_clean)
        st.success("Data berhasil dimuat ✅")
        st.caption(f"📊 {len(df):,} baris • {len(df.columns)} kolom".replace(",","."))

        date_range = tahun_pilihan = poli_pilihan = jk_pilihan = None
        bayar_pilihan = kelompok_umur_pilihan = desa_pilihan = kecuali_penyakit_pilihan = None

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
                    help="Penyakit yang dipilih TIDAK AKAN diikutkan dalam analisis."
                )

    df_filtered = df.copy()
    if date_range and len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered["tanggal_kunjungan"].dt.date >= date_range[0]) &
            (df_filtered["tanggal_kunjungan"].dt.date <= date_range[1])
        ]
    if tahun_pilihan:             df_filtered = df_filtered[df_filtered["tahun"].isin(tahun_pilihan)]
    if poli_pilihan:              df_filtered = df_filtered[df_filtered["poli"].isin(poli_pilihan)]
    if jk_pilihan:                df_filtered = df_filtered[df_filtered["jenis_kelamin"].isin(jk_pilihan)]
    if bayar_pilihan:             df_filtered = df_filtered[df_filtered["pembiayaan"].isin(bayar_pilihan)]
    if kelompok_umur_pilihan:     df_filtered = df_filtered[df_filtered["kelompok_umur"].isin(kelompok_umur_pilihan)]
    if desa_pilihan:              df_filtered = df_filtered[df_filtered["desa"].isin(desa_pilihan)]
    if kecuali_penyakit_pilihan:  df_filtered = df_filtered[~df_filtered["diagnosa"].isin(kecuali_penyakit_pilihan)]

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
            chips.append(f"{k.replace('_',' ').title()}: {', '.join(map(str,v))}")
    if chips: st.caption("🎯 **Filter aktif:** " + " | ".join(chips))

# ========= HALAMAN OVERVIEW =========
def page_overview(df_filtered, filter_info):
    st.subheader("📌 Ringkasan Umum")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data."); return
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Kunjungan", len(df_filtered))
    if "no_rm"    in df_filtered.columns: col2.metric("Pasien Unik",  df_filtered["no_rm"].nunique())
    if "poli"     in df_filtered.columns: col3.metric("Poli Aktif",   df_filtered["poli"].nunique())
    if "diagnosa" in df_filtered.columns: col4.metric("Diagnosa",     df_filtered["diagnosa"].nunique())
    st.markdown("---")
    if "tanggal_kunjungan" in df_filtered.columns:
        st.markdown("### 📈 Tren Kunjungan")
        trend = df_filtered.groupby(["tahun","bulan","nama_bulan"]).size().reset_index(name="count").sort_values(["tahun","bulan"])
        trend["label"] = trend["nama_bulan"].astype(str)+"-"+trend["tahun"].astype(str)
        st.line_chart(trend.set_index("label")["count"])
        st.download_button("📥 Download Tren Kunjungan (Excel)", convert_df_to_excel(trend), "tren_kunjungan.xlsx")
    if "poli" in df_filtered.columns:
        st.markdown("### 🏥 Distribusi Poli")
        df_poli = df_filtered["poli"].value_counts().reset_index()
        df_poli.columns = ["Poli","Jumlah"]
        st.bar_chart(df_poli.set_index("Poli"))
        st.download_button("📥 Download Distribusi Poli (Excel)", convert_df_to_excel(df_poli), "distribusi_poli.xlsx")

# ========= HALAMAN KUNJUNGAN =========
def page_kunjungan(df_filtered, filter_info):
    st.subheader("👥 Analisis Kunjungan")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    col1, col2 = st.columns(2)
    if "jenis_kelamin" in df_filtered.columns:
        col1.markdown("#### Jenis Kelamin")
        df_jk = df_filtered["jenis_kelamin"].value_counts().reset_index()
        df_jk.columns = ["Jenis Kelamin","Jumlah"]
        col1.bar_chart(df_jk.set_index("Jenis Kelamin"))
        col1.download_button("📥 Download Data Gender", convert_df_to_excel(df_jk), "kunjungan_gender.xlsx")
    if "kelompok_umur" in df_filtered.columns:
        col2.markdown("#### Kelompok Umur")
        df_umur = df_filtered["kelompok_umur"].value_counts().sort_index().reset_index()
        df_umur.columns = ["Kelompok Umur","Jumlah"]
        col2.bar_chart(df_umur.set_index("Kelompok Umur"))
        col2.download_button("📥 Download Data Umur", convert_df_to_excel(df_umur), "kunjungan_umur.xlsx")

# ========= HALAMAN PENYAKIT =========
def page_penyakit(df_filtered, filter_info):
    st.subheader("🦠 Analisis Penyakit")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    if "diagnosa" not in df_filtered.columns:
        st.error("❌ Kolom 'diagnosa' tidak ditemukan."); return

    col_slider, col_urutan, col_tampilan = st.columns([2,2,1])
    with col_slider:
        top_n = st.slider("Jumlah diagnosa ditampilkan", 5, 30, 10, 1)
    with col_urutan:
        urutan = st.radio("Urutkan:", ["⬇️ Terbanyak → Tersedikit","⬆️ Tersedikit → Terbanyak"],
                          index=0, horizontal=True, key="radio_urutan_penyakit")
    with col_tampilan:
        orientasi = st.selectbox("Orientasi grafik", ["Horizontal","Vertikal"],
                                  index=0, key="select_orientasi_penyakit")

    ascending = urutan.startswith("⬆️")
    df_diag = df_filtered["diagnosa"].value_counts().head(top_n).reset_index()
    df_diag.columns = ["Diagnosa","Jumlah Kasus"]
    df_diag = df_diag.sort_values("Jumlah Kasus", ascending=ascending).reset_index(drop=True)
    df_diag["Peringkat"] = df_diag["Jumlah Kasus"].rank(ascending=False, method="min").astype(int)
    color_scale  = "Blues" if not ascending else "Blues_r"
    chart_height = max(350, top_n * 38)

    if orientasi == "Horizontal":
        fig = px.bar(df_diag, x="Jumlah Kasus", y="Diagnosa", orientation="h",
                     text="Jumlah Kasus", color="Jumlah Kasus",
                     color_continuous_scale=color_scale, custom_data=["Peringkat"])
        fig.update_traces(textposition="outside",
                          hovertemplate="<b>%{y}</b><br>Jumlah Kasus: <b>%{x}</b><br>Peringkat: #%{customdata[0]}<extra></extra>")
        fig.update_layout(yaxis=dict(categoryorder="total ascending" if ascending else "total descending"),
                          coloraxis_showscale=False, margin=dict(l=0,r=70,t=10,b=10), height=chart_height)
    else:
        fig = px.bar(df_diag, x="Diagnosa", y="Jumlah Kasus", orientation="v",
                     text="Jumlah Kasus", color="Jumlah Kasus",
                     color_continuous_scale=color_scale, custom_data=["Peringkat"])
        fig.update_traces(textposition="outside",
                          hovertemplate="<b>%{x}</b><br>Jumlah Kasus: <b>%{y}</b><br>Peringkat: #%{customdata[0]}<extra></extra>")
        fig.update_layout(xaxis=dict(categoryorder="total descending" if not ascending else "total ascending",
                                     tickangle=-35),
                          coloraxis_showscale=False, margin=dict(l=10,r=10,t=30,b=10), height=chart_height)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    total_kasus_tampil = df_diag["Jumlah Kasus"].sum()
    total_semua_kasus  = len(df_filtered)
    pct = (total_kasus_tampil / total_semua_kasus * 100) if total_semua_kasus > 0 else 0
    top_row = df_diag.sort_values("Jumlah Kasus", ascending=False).iloc[0]
    m1,m2,m3 = st.columns(3)
    m1.metric("Total Kasus (ditampilkan)", f"{total_kasus_tampil:,}".replace(",","."))
    m2.metric("% dari Seluruh Kunjungan",  f"{pct:.1f}%")
    m3.metric("Diagnosa Teratas", top_row["Diagnosa"],
              delta=f"{top_row['Jumlah Kasus']} kasus", delta_color="off")

    with st.expander("📋 Lihat Tabel Detail", expanded=False):
        df_tabel = df_diag[["Peringkat","Diagnosa","Jumlah Kasus"]].copy()
        df_tabel["% dari Total"] = (df_tabel["Jumlah Kasus"]/total_semua_kasus*100).round(2).astype(str)+"%"
        st.dataframe(df_tabel, use_container_width=True, hide_index=True)
    st.download_button("📥 Download Data Top Penyakit (Excel)",
                        convert_df_to_excel(df_diag[["Peringkat","Diagnosa","Jumlah Kasus"]]),
                        "top_penyakit.xlsx")

# ========= HALAMAN PETA =========
def page_peta_persebaran(df_filtered, filter_info):
    st.subheader("🗺️ Peta Persebaran Penyakit")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong."); return
    if "desa" not in df_filtered.columns or "diagnosa" not in df_filtered.columns:
        st.error("❌ Kolom 'desa' atau 'diagnosa' tidak ditemukan."); return

    col_pilih_penyakit, col_pilih_tema = st.columns([3,1])
    with col_pilih_penyakit:
        top_penyakit     = df_filtered["diagnosa"].value_counts().head(20).index.tolist()
        pilihan_penyakit = st.selectbox("Pilihan Diagnosa:", options=["-- Semua Penyakit (Top 10) --"]+top_penyakit)
    with col_pilih_tema:
        tema_peta = st.selectbox("Tema Peta:", ["Terang","Gelap"])

    map_style = "carto-darkmatter" if tema_peta == "Gelap" else "carto-positron"
    if pilihan_penyakit != "-- Semua Penyakit (Top 10) --":
        df_map = df_filtered[df_filtered["diagnosa"] == pilihan_penyakit].copy()
    else:
        top_10 = df_filtered["diagnosa"].value_counts().head(10).index.tolist()
        df_map = df_filtered[df_filtered["diagnosa"].isin(top_10)].copy()

    df_grouped = df_map.groupby(["desa","diagnosa"]).size().reset_index(name="jumlah_kasus")
    koordinat_desa = {
        "Donan":(-7.2131,111.6364),"Gapluk":(-7.2017,111.6617),"Kaliombo":(-7.235,111.6817),
        "Kuniran":(-7.234645,111.651014),"Ngrejeng":(-7.22604,111.70707),"Pelem":(-7.2394,111.7011),
        "Pojok":(-7.1892,111.6728),"Punggur":(-7.2048,111.6808),"Purwosari":(-7.1798,111.6608),
        "Sedahkidul":(-7.1973,111.6792),"Tinumpuk":(-7.2117,111.68),"Tlatah":(-7.2172,111.6975)
    }
    def get_koordinat(nama_desa):
        return koordinat_desa.get(str(nama_desa).strip().title(),(-7.1509,111.8817))

    df_grouped["latitude"]  = df_grouped["desa"].apply(lambda x: get_koordinat(x)[0])
    df_grouped["longitude"] = df_grouped["desa"].apply(lambda x: get_koordinat(x)[1])
    desa_agregat    = df_grouped.groupby("desa")["jumlah_kasus"].sum()
    desa_tertinggi  = desa_agregat.idxmax() if not desa_agregat.empty else "-"
    kasus_tertinggi = desa_agregat.max()    if not desa_agregat.empty else 0
    col1,col2,col3 = st.columns(3)
    col1.metric("Total Kasus Terpeta",  f"{df_grouped['jumlah_kasus'].sum()} Pasien")
    col2.metric("Total Desa Terdampak", f"{df_grouped['desa'].nunique()} Desa")
    col3.metric("Desa Kasus Tertinggi", desa_tertinggi, delta=f"{kasus_tertinggi} Kasus", delta_color="off")

    fig = px.scatter_mapbox(df_grouped, lat="latitude", lon="longitude", color="diagnosa",
                            size="jumlah_kasus", hover_name="desa", custom_data=["desa"],
                            hover_data={"latitude":False,"longitude":False,"diagnosa":True,
                                        "jumlah_kasus":True,"desa":False},
                            color_discrete_sequence=px.colors.qualitative.Plotly,
                            zoom=11.5, center={"lat":-7.218,"lon":111.675}, height=550, size_max=35)
    fig.update_layout(mapbox_style=map_style, margin={"r":0,"t":0,"l":0,"b":0})
    clicked_desa = None
    try:
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        if hasattr(event,"selection") and hasattr(event.selection,"points") and len(event.selection.points)>0:
            pt = event.selection.points[0]
            clicked_desa = pt.get("customdata",[None])[0] or pt.get("hovertext")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)
        fb = st.selectbox("Pilih Desa:", options=["-- Klik Pilih --"]+
                          sorted(df_filtered["desa"].dropna().unique().tolist()))
        if fb != "-- Klik Pilih --": clicked_desa = fb

    st.info("👆 **TIPS:** Klik pada titik desa di peta untuk melihat profil kesehatan desa tersebut!")
    if clicked_desa:
        st.markdown(f"### 📍 Profil Kesehatan Desa: **{clicked_desa}**")
        df_desa = df_filtered[df_filtered["desa"] == clicked_desa]
        if len(df_desa) > 0:
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Kunjungan", f"{len(df_desa)} Pasien")
            c2.metric("Penyakit Dominan", df_desa["diagnosa"].mode()[0] if "diagnosa" in df_desa.columns else "-")
            if "jenis_kelamin" in df_desa.columns:
                c3.metric("Mayoritas Gender", df_desa["jenis_kelamin"].mode()[0])
            col_d1,col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Top 5 Penyakit**")
                st.bar_chart(df_desa["diagnosa"].value_counts().head(5))
            with col_d2:
                st.markdown("**Demografi Usia**")
                if "kelompok_umur" in df_desa.columns:
                    st.bar_chart(df_desa["kelompok_umur"].value_counts().sort_index())
    st.markdown("---")
    col_t,col_g = st.columns(2)
    with col_t:
        st.markdown("#### 📋 Detail Kasus per Desa")
        st.dataframe(df_grouped[["desa","diagnosa","jumlah_kasus"]].sort_values("jumlah_kasus",ascending=False),
                     use_container_width=True, hide_index=True)
    with col_g:
        st.markdown("#### 📊 Top Desa Terdampak")
        top_desa_df = (df_grouped.groupby("desa")["jumlah_kasus"].sum()
                       .reset_index().sort_values("jumlah_kasus",ascending=False).head(10))
        st.bar_chart(top_desa_df.set_index("desa")["jumlah_kasus"])
        st.download_button("📥 Download Top Desa (Excel)", convert_df_to_excel(top_desa_df),
                            "top_desa_terdampak.xlsx")

# ========= HALAMAN PEMBIAYAAN =========
def page_pembiayaan(df_filtered, filter_info):
    st.subheader("💳 Analisis Pembiayaan")
    if df_filtered is None or "pembiayaan" not in df_filtered.columns: return
    df_bayar = df_filtered["pembiayaan"].value_counts().reset_index()
    df_bayar.columns = ["Pembiayaan","Jumlah"]
    st.bar_chart(df_bayar.set_index("Pembiayaan"))
    st.download_button("📥 Download Data Pembiayaan (Excel)", convert_df_to_excel(df_bayar), "pembiayaan.xlsx")

# =============================================================================
# ██╗  ██╗ ███████╗ ██╗       ██████╗  ██████╗  ██████╗ ██╗ ███╗   ███╗
# ██║ ██╔╝ ██╔════╝ ██║      ██╔═══██╗██╔════╝ ██╔════╝ ██║ ████╗ ████║
# █████╔╝  █████╗   ██║      ██║   ██║██║  ███╗██║  ███╗██║ ██╔████╔██║
# ██╔═██╗  ██╔══╝   ██║      ██║   ██║██║   ██║██║   ██║██║ ██║╚██╔╝██║
# ██║  ██╗ ███████╗ ███████╗ ╚██████╔╝╚██████╔╝╚██████╔╝██║ ██║ ╚═╝ ██║
# ╚═╝  ╚═╝ ╚══════╝ ╚══════╝  ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝ ╚═╝     ╚═╝
# ===== HALAMAN ML TEROPTIMASI ================================================
# =============================================================================

# ── Hari Libur Nasional Indonesia 2022-2026 ───────────────────────────────────
def _get_holidays_indonesia() -> pd.DataFrame:
    records = [
        # 2022
        ("2022-01-01","Tahun Baru Masehi"),("2022-02-01","Tahun Baru Imlek"),
        ("2022-03-03","Isra Miraj"),("2022-04-15","Wafat Isa Almasih"),
        ("2022-05-01","Hari Buruh"),("2022-05-03","Hari Raya Idul Fitri"),
        ("2022-05-04","Hari Raya Idul Fitri"),("2022-05-16","Hari Raya Waisak"),
        ("2022-05-26","Kenaikan Isa Almasih"),("2022-06-01","Hari Lahir Pancasila"),
        ("2022-07-10","Hari Raya Idul Adha"),("2022-07-30","Tahun Baru Islam"),
        ("2022-08-17","Hari Kemerdekaan RI"),("2022-10-08","Maulid Nabi Muhammad"),
        ("2022-12-25","Hari Raya Natal"),
        # 2023
        ("2023-01-01","Tahun Baru Masehi"),("2023-01-22","Tahun Baru Imlek"),
        ("2023-02-18","Isra Miraj"),("2023-03-22","Hari Suci Nyepi"),
        ("2023-04-07","Wafat Isa Almasih"),("2023-04-22","Hari Raya Idul Fitri"),
        ("2023-04-23","Hari Raya Idul Fitri"),("2023-05-01","Hari Buruh"),
        ("2023-05-18","Kenaikan Isa Almasih"),("2023-06-01","Hari Lahir Pancasila"),
        ("2023-06-02","Hari Raya Waisak"),("2023-06-29","Hari Raya Idul Adha"),
        ("2023-07-19","Tahun Baru Islam"),("2023-08-17","Hari Kemerdekaan RI"),
        ("2023-09-28","Maulid Nabi Muhammad"),("2023-12-25","Hari Raya Natal"),
        # 2024
        ("2024-01-01","Tahun Baru Masehi"),("2024-02-08","Tahun Baru Imlek"),
        ("2024-02-14","Pemilu"),("2024-03-11","Isra Miraj / Hari Suci Nyepi"),
        ("2024-03-29","Wafat Isa Almasih"),("2024-04-10","Hari Raya Idul Fitri"),
        ("2024-04-11","Hari Raya Idul Fitri"),("2024-05-01","Hari Buruh"),
        ("2024-05-09","Kenaikan Isa Almasih"),("2024-05-23","Hari Raya Waisak"),
        ("2024-06-01","Hari Lahir Pancasila"),("2024-06-17","Hari Raya Idul Adha"),
        ("2024-07-07","Tahun Baru Islam"),("2024-08-17","Hari Kemerdekaan RI"),
        ("2024-09-16","Maulid Nabi Muhammad"),("2024-12-25","Hari Raya Natal"),
        # 2025
        ("2025-01-01","Tahun Baru Masehi"),("2025-01-29","Tahun Baru Imlek"),
        ("2025-03-28","Hari Suci Nyepi"),("2025-03-29","Isra Miraj"),
        ("2025-03-30","Hari Raya Idul Fitri"),("2025-03-31","Hari Raya Idul Fitri"),
        ("2025-04-18","Wafat Isa Almasih"),("2025-05-01","Hari Buruh"),
        ("2025-05-12","Hari Raya Waisak"),("2025-05-29","Kenaikan Isa Almasih"),
        ("2025-06-01","Hari Lahir Pancasila"),("2025-06-07","Hari Raya Idul Adha"),
        ("2025-06-27","Tahun Baru Islam"),("2025-08-17","Hari Kemerdekaan RI"),
        ("2025-09-05","Maulid Nabi Muhammad"),("2025-12-25","Hari Raya Natal"),
        # 2026
        ("2026-01-01","Tahun Baru Masehi"),("2026-02-17","Tahun Baru Imlek"),
        ("2026-03-17","Hari Suci Nyepi"),("2026-03-19","Isra Miraj"),
        ("2026-03-20","Hari Raya Idul Fitri"),("2026-03-21","Hari Raya Idul Fitri"),
        ("2026-04-02","Wafat Isa Almasih"),("2026-05-01","Hari Buruh"),
        ("2026-05-14","Kenaikan Isa Almasih"),("2026-05-27","Hari Raya Idul Adha"),
        ("2026-05-31","Hari Raya Waisak"),("2026-06-01","Hari Lahir Pancasila"),
        ("2026-06-17","Tahun Baru Islam"),("2026-08-17","Hari Kemerdekaan RI"),
        ("2026-08-27","Maulid Nabi Muhammad"),("2026-12-25","Hari Raya Natal"),
    ]
    df_h = pd.DataFrame(records, columns=["ds","holiday"])
    df_h["ds"] = pd.to_datetime(df_h["ds"])
    df_h = df_h.drop_duplicates(subset=["ds"])
    # Efek liburan mencakup H-1 dan H+1
    df_h["lower_window"] = -1
    df_h["upper_window"] = 1
    return df_h

# ── Deteksi & Imputasi Outlier (IQR-based) ────────────────────────────────────
def _remove_outliers_iqr(df_prophet: pd.DataFrame, multiplier: float = 1.5):
    df = df_prophet.copy()
    Q1, Q3 = df["y"].quantile(0.25), df["y"].quantile(0.75)
    IQR    = Q3 - Q1
    lower, upper = Q1 - multiplier*IQR, Q3 + multiplier*IQR
    is_outlier   = (df["y"] < lower) | (df["y"] > upper)
    n_outlier    = int(is_outlier.sum())
    if n_outlier > 0:
        rolling_med = df["y"].rolling(window=4, min_periods=1, center=True).median()
        df.loc[is_outlier, "y"] = rolling_med[is_outlier]
    df["y"] = df["y"].clip(lower=0)
    return df, n_outlier, float(lower), float(upper)

# ── Pilih frekuensi agregasi optimal ─────────────────────────────────────────
def _pilih_frekuensi(total_days: int, n_rows: int) -> tuple:
    if total_days >= 180 and n_rows >= 60:
        return "D", "Harian"
    elif total_days >= 56:
        return "W-MON", "Mingguan"
    else:
        return "MS", "Bulanan"

# ── 5 Kandidat Hyperparameter ─────────────────────────────────────────────────
PARAM_GRID = [
    (0.01,  1.0, "Konservatif (CPS=0.01)"),
    (0.05,  5.0, "Standar (CPS=0.05)"),
    (0.10,  5.0, "Moderat (CPS=0.10)"),
    (0.30, 10.0, "Fleksibel (CPS=0.30)"),
    (0.50, 10.0, "Sangat Fleksibel (CPS=0.50)"),
]

def _build_prophet_params(cps: float, sps: float, freq: str) -> dict:
    return dict(
        yearly_seasonality        = True,
        weekly_seasonality        = (freq == "D"),
        daily_seasonality         = False,
        changepoint_prior_scale   = cps,
        seasonality_prior_scale   = sps,
        interval_width            = 0.80,
    )

def _render_akurasi_badge(mape_avg: float) -> str:
    if mape_avg < 10:
        return f'<span class="akurasi-badge akurasi-hijau">🟢 Sangat Baik — MAPE {mape_avg:.1f}%</span>'
    elif mape_avg < 20:
        return f'<span class="akurasi-badge akurasi-kuning">🟡 Cukup Baik — MAPE {mape_avg:.1f}%</span>'
    elif mape_avg < 30:
        return f'<span class="akurasi-badge akurasi-oranye">🟠 Perlu Perhatian — MAPE {mape_avg:.1f}%</span>'
    else:
        return f'<span class="akurasi-badge akurasi-merah">🔴 Kurang Akurat — MAPE {mape_avg:.1f}%</span>'

# ── Jalankan CV untuk satu set parameter ─────────────────────────────────────
def _run_single_cv(params: dict, df_clean: pd.DataFrame,
                   initial_str: str, period_str: str, horizon_str: str):
    holidays = _get_holidays_indonesia()
    m = Prophet(**params)
    m.holidays = holidays
    m.fit(df_clean)
    df_cv = cross_validation(m, initial=initial_str, period=period_str,
                              horizon=horizon_str, parallel=None)
    return performance_metrics(df_cv)

# ── Latih model final ─────────────────────────────────────────────────────────
def _train_final_model(df_clean: pd.DataFrame, params: dict,
                       periods_ahead: int, freq: str):
    holidays = _get_holidays_indonesia()
    m = Prophet(**params)
    m.holidays = holidays
    m.fit(df_clean)
    future   = m.make_future_dataframe(periods=periods_ahead, freq=freq)
    forecast = m.predict(future)
    return m, forecast

# ── Section evaluasi + auto-tuning ────────────────────────────────────────────
def section_evaluasi_akurasi_optimal(df_prophet_raw: pd.DataFrame,
                                     df_prophet_clean: pd.DataFrame,
                                     freq: str, n_outlier: int,
                                     outlier_lower: float, outlier_upper: float):
    with st.expander("📐 Evaluasi Akurasi & Auto-Tuning Hyperparameter", expanded=False):

        # ── Info pra-pemrosesan ────────────────────────────────────────────
        total_days = (df_prophet_clean["ds"].max() - df_prophet_clean["ds"].min()).days
        freq_label = {"D":"Harian","W-MON":"Mingguan","MS":"Bulanan"}.get(freq, freq)

        st.markdown("#### ⚙️ Pra-Pemrosesan Data Time-Series")
        pre1, pre2, pre3 = st.columns(3)
        pre1.metric("Frekuensi Agregasi",            freq_label)
        pre2.metric("Outlier Terdeteksi & Diperbaiki", f"{n_outlier} titik")
        pre3.metric("Rentang Data",                  f"{total_days} hari")

        if n_outlier > 0:
            st.info(f"ℹ️ **{n_outlier} outlier** di luar rentang [{outlier_lower:.1f}–{outlier_upper:.1f}] "
                    f"diganti nilai median rolling 4 periode.")
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=df_prophet_raw["ds"], y=df_prophet_raw["y"],
                                          mode="lines+markers", name="Data Asli",
                                          line=dict(color="#94a3b8",width=1.5,dash="dot")))
            fig_cmp.add_trace(go.Scatter(x=df_prophet_clean["ds"], y=df_prophet_clean["y"],
                                          mode="lines+markers", name="Setelah Koreksi",
                                          line=dict(color="#2563eb",width=2)))
            fig_cmp.update_layout(title="Data Sebelum & Sesudah Koreksi Outlier",
                                   hovermode="x unified", margin=dict(t=40,b=20))
            st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("---")

        if total_days < 60:
            st.warning("⚠️ Data terlalu pendek untuk cross-validation (minimal ~60 hari).")
            return None

        # ── Parameter CV ───────────────────────────────────────────────────
        initial_days = max(30, int(total_days * 0.5))
        period_days  = max(14, int(total_days * 0.15))
        horizon_days = max(14, int(total_days * 0.25))
        initial_str  = f"{initial_days} days"
        period_str   = f"{period_days} days"
        horizon_str  = f"{horizon_days} days"

        st.markdown("#### 🔬 Auto-Tuning: Menguji 5 Kandidat Hyperparameter")
        st.caption("Sistem mencari kombinasi parameter terbaik menggunakan walk-forward cross-validation. "
                   "Model dengan **MAPE terkecil** dipilih sebagai model final prediksi.")

        pc1,pc2,pc3 = st.columns(3)
        pc1.info(f"⚙️ **Initial**\n\n{initial_str}")
        pc2.info(f"📅 **Period**\n\n{period_str}")
        pc3.info(f"🔭 **Horizon**\n\n{horizon_str}")

        # ── Loop auto-tuning ───────────────────────────────────────────────
        results   = []
        prog_bar  = st.progress(0, text="Memulai auto-tuning…")
        log_area  = st.empty()
        log_lines = []

        for idx, (cps, sps, label) in enumerate(PARAM_GRID):
            pct = int(((idx) / len(PARAM_GRID)) * 100)
            prog_bar.progress(pct, text=f"⏳ Menguji: {label} ({idx+1}/{len(PARAM_GRID)})")
            log_lines.append(f"🔄 [{idx+1}/{len(PARAM_GRID)}] Menguji **{label}**…")
            log_area.markdown("\n\n".join(log_lines))
            try:
                params  = _build_prophet_params(cps, sps, freq)
                df_pm   = _run_single_cv(params, df_prophet_clean, initial_str, period_str, horizon_str)
                mape    = df_pm["mape"].mean() * 100
                mae     = df_pm["mae"].mean()
                rmse    = df_pm["rmse"].mean()
                cov     = df_pm["coverage"].mean() * 100
                results.append({"label":label,"cps":cps,"sps":sps,
                                 "mape":mape,"mae":mae,"rmse":rmse,"coverage":cov,
                                 "df_pm":df_pm,"params":params})
                log_lines[-1] = f"✅ [{idx+1}/{len(PARAM_GRID)}] **{label}** → MAPE: **{mape:.1f}%**"
            except Exception as e:
                log_lines[-1] = f"⚠️ [{idx+1}/{len(PARAM_GRID)}] **{label}** → Gagal: {e}"
            log_area.markdown("\n\n".join(log_lines))

        prog_bar.progress(100, text="✅ Auto-tuning selesai!")

        if not results:
            st.error("❌ Semua kandidat gagal. Periksa kecukupan data.")
            return None

        results.sort(key=lambda x: x["mape"])
        best = results[0]

        st.markdown("---")
        st.markdown("#### 🏆 Perbandingan Semua Kandidat Model")
        for r in results:
            is_best  = (r["label"] == best["label"])
            card_cls = "model-card model-card-best" if is_best else "model-card model-card-normal"
            badge    = " 🏆 **TERPILIH**" if is_best else ""
            st.markdown(
                f'<div class="{card_cls}">'
                f'<b>{r["label"]}</b>{badge}<br>'
                f'MAPE: <b>{r["mape"]:.1f}%</b> &nbsp;|&nbsp; '
                f'MAE: {r["mae"]:.1f} &nbsp;|&nbsp; '
                f'RMSE: {r["rmse"]:.1f} &nbsp;|&nbsp; '
                f'Coverage: {r["coverage"]:.1f}%'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(f"#### 📊 Metrik Detail Model Terbaik: *{best['label']}*")
        st.markdown(_render_akurasi_badge(best["mape"]), unsafe_allow_html=True)
        st.markdown("")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("MAE",      f"{best['mae']:.1f}",      help="Rata-rata selisih absolut prediksi vs aktual")
        m2.metric("RMSE",     f"{best['rmse']:.1f}",     help="Lebih sensitif terhadap error besar")
        m3.metric("MAPE",     f"{best['mape']:.1f}%",    help="Error dalam persen — makin kecil makin akurat")
        m4.metric("Coverage", f"{best['coverage']:.1f}%",help="% data aktual dalam rentang prediksi (target ≈ 80%)")

        st.markdown("""
            **Panduan MAPE:**
            - 🟢 **< 10%** → Sangat akurat, layak jadi acuan keputusan
            - 🟡 **10–20%** → Cukup baik; wajar untuk data kunjungan puskesmas
            - 🟠 **20–30%** → Perlu dicermati; data mungkin terlalu sedikit
            - 🔴 **> 30%** → Kurang andal; tambahkan data historis lebih banyak
        """)

        st.markdown("---")
        st.markdown("#### 📉 Tren Akurasi vs Jarak Prediksi (Horizon)")
        st.caption("Semakin jauh horizon prediksi, umumnya akurasi menurun — perilaku normal semua model time-series.")

        df_pm_best = best["df_pm"].copy()
        df_pm_best["horizon_hari"] = df_pm_best["horizon"].dt.days
        df_pm_best["mape_pct"]     = df_pm_best["mape"] * 100
        df_pm_best["coverage_pct"] = df_pm_best["coverage"] * 100

        tab_mape, tab_mae, tab_cov, tab_compare = st.tabs(
            ["MAPE (%)","MAE & RMSE","Coverage (%)","Perbandingan Semua Model"]
        )

        with tab_mape:
            fig_m = px.line(df_pm_best, x="horizon_hari", y="mape_pct", markers=True,
                            labels={"horizon_hari":"Horizon (hari)","mape_pct":"MAPE (%)"},
                            title=f"MAPE vs Horizon — {best['label']}")
            fig_m.add_hline(y=10, line_dash="dash", line_color="green",  annotation_text="Sangat Baik (10%)")
            fig_m.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Cukup Baik (20%)")
            fig_m.add_hline(y=30, line_dash="dash", line_color="red",    annotation_text="Kurang Akurat (30%)")
            fig_m.update_layout(yaxis_ticksuffix="%", margin=dict(t=40,b=20))
            st.plotly_chart(fig_m, use_container_width=True)

        with tab_mae:
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=df_pm_best["horizon_hari"], y=df_pm_best["mae"],
                                        mode="lines+markers", name="MAE",
                                        line=dict(color="#2563eb",width=2)))
            fig_e.add_trace(go.Scatter(x=df_pm_best["horizon_hari"], y=df_pm_best["rmse"],
                                        mode="lines+markers", name="RMSE",
                                        line=dict(color="#ef4444",width=2,dash="dash")))
            fig_e.update_layout(title=f"MAE & RMSE vs Horizon — {best['label']}",
                                 xaxis_title="Horizon (hari)", yaxis_title="Error (pasien)",
                                 hovermode="x unified", margin=dict(t=40,b=20))
            st.plotly_chart(fig_e, use_container_width=True)

        with tab_cov:
            fig_c = px.line(df_pm_best, x="horizon_hari", y="coverage_pct", markers=True,
                            labels={"horizon_hari":"Horizon (hari)","coverage_pct":"Coverage (%)"},
                            title=f"Coverage vs Horizon — {best['label']}")
            fig_c.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target 80%")
            fig_c.update_layout(yaxis_ticksuffix="%", margin=dict(t=40,b=20))
            st.plotly_chart(fig_c, use_container_width=True)

        with tab_compare:
            df_cmp = pd.DataFrame([{"Model":r["label"],"MAPE (%)":round(r["mape"],2),
                                     "MAE":round(r["mae"],2),"RMSE":round(r["rmse"],2),
                                     "Coverage (%)":round(r["coverage"],2)} for r in results])
            fig_bar = px.bar(df_cmp, x="Model", y="MAPE (%)", text="MAPE (%)",
                              color="MAPE (%)", color_continuous_scale="RdYlGn_r",
                              title="Perbandingan MAPE Antar Kandidat Model")
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(coloraxis_showscale=False, margin=dict(t=40,b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Perbandingan Model (Excel)",
                                convert_df_to_excel(df_cmp), "perbandingan_model.xlsx")

        with st.expander("📋 Tabel Detail Metrik Model Terbaik per Horizon"):
            df_d = df_pm_best[["horizon_hari","mae","rmse","mape_pct","coverage_pct"]].copy()
            df_d.columns = ["Horizon (hari)","MAE","RMSE","MAPE (%)","Coverage (%)"]
            df_d["MAPE (%)"]     = df_d["MAPE (%)"].round(2).astype(str)+"%"
            df_d["Coverage (%)"] = df_d["Coverage (%)"].round(2).astype(str)+"%"
            st.dataframe(df_d, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Metrik Akurasi (Excel)",
                                convert_df_to_excel(df_d), "evaluasi_akurasi.xlsx")

        return best["params"]


# ================== HALAMAN ML TEROPTIMASI ====================================
def page_ml(df_filtered, filter_info):
    st.subheader("📈 Prediksi & Peramalan Tren — Model Teroptimasi")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong."); return
    if "tanggal_kunjungan" not in df_filtered.columns:
        st.error("❌ Kolom 'tanggal_kunjungan' tidak ditemukan."); return

    df_ml = df_filtered.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_ml["tanggal_kunjungan"]):
        df_ml["tanggal_kunjungan"] = pd.to_datetime(df_ml["tanggal_kunjungan"], errors="coerce")
    df_ml = df_ml.dropna(subset=["tanggal_kunjungan"])
    if len(df_ml) == 0:
        st.warning("⚠️ Data tanggal tidak valid."); return

    max_date = df_ml["tanggal_kunjungan"].max().date()

    st.markdown(
        '<div class="opt-banner">'
        "🚀 <b>Mode Teroptimasi Aktif:</b> Dashboard secara otomatis menjalankan "
        "<b>deteksi & koreksi outlier</b>, <b>pemilihan frekuensi terbaik</b>, "
        "<b>penambahan hari libur nasional Indonesia (2022–2026)</b>, dan "
        "<b>auto-tuning 5 kandidat hyperparameter</b> untuk mencari model paling akurat."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col_s1, col_s2, col_s3 = st.columns([1,1,1])
    with col_s1:
        fokus       = st.radio("Analisis:", ["Diagnosa Penyakit","Poli / Unit"], horizontal=True)
        kolom_fokus = "diagnosa" if fokus == "Diagnosa Penyakit" else "poli"
    with col_s2:
        if kolom_fokus not in df_ml.columns: return
        top_items    = df_ml[kolom_fokus].value_counts().head(30)
        pilihan_item = st.selectbox(f"Pilih {fokus}:", options=top_items.index.tolist())
    with col_s3:
        target_date = st.date_input(
            "Prediksi Sampai Tanggal:",
            value     = max_date + pd.Timedelta(days=30),
            min_value = max_date + pd.Timedelta(days=1),
            max_value = max_date + pd.Timedelta(days=365)
        )

    # ── Opsi Lanjutan ──────────────────────────────────────────────────────
    with st.expander("⚙️ Opsi Lanjutan (Opsional)", expanded=False):
        iqr_multiplier = st.slider(
            "Sensitivitas Deteksi Outlier (IQR Multiplier)",
            min_value=1.0, max_value=3.0, value=1.5, step=0.1,
            help="Nilai lebih kecil = lebih sensitif. Default 1.5 (standar statistik)."
        )
        manual_mode = st.checkbox(
            "Mode Manual: Nonaktifkan Auto-Tuning (pakai parameter kustom)",
            value=False,
            help="Centang untuk melewati auto-tuning dan mempercepat loading."
        )
        if manual_mode:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                manual_cps = st.select_slider(
                    "changepoint_prior_scale",
                    options=[0.01,0.05,0.10,0.30,0.50], value=0.05,
                    help="Fleksibilitas tren. Besar = lebih fleksibel."
                )
            with col_m2:
                manual_sps = st.select_slider(
                    "seasonality_prior_scale",
                    options=[1.0,5.0,10.0,15.0], value=5.0,
                    help="Kekuatan musiman. Besar = musiman lebih dominan."
                )

    # ── Persiapan data ─────────────────────────────────────────────────────
    df_item = df_ml[df_ml[kolom_fokus] == pilihan_item].copy()
    if len(df_item) < 10:
        st.error("❌ Data terlalu sedikit (< 10 pasien) untuk dilatih model."); return

    total_days = (df_item["tanggal_kunjungan"].max() - df_item["tanggal_kunjungan"].min()).days
    freq, freq_label = _pilih_frekuensi(total_days, len(df_item))

    agg = df_item.groupby(pd.Grouper(key="tanggal_kunjungan", freq=freq)).size().reset_index(name="jumlah")
    df_prophet_raw   = agg.rename(columns={"tanggal_kunjungan":"ds","jumlah":"y"})
    df_prophet_raw   = df_prophet_raw[df_prophet_raw["y"] > 0].reset_index(drop=True)
    df_prophet_clean, n_outlier, out_lower, out_upper = _remove_outliers_iqr(
        df_prophet_raw, multiplier=iqr_multiplier
    )

    with st.expander("👁️ Lihat Data yang Dipelajari Model", expanded=False):
        st.caption(f"Frekuensi dipilih otomatis: **{freq_label}** ({total_days} hari data)")
        col_r, col_c = st.columns(2)
        col_r.markdown("**Data Asli**")
        col_r.dataframe(df_prophet_raw.rename(columns={"ds":"Periode","y":"Jumlah"}), use_container_width=True)
        col_c.markdown("**Setelah Koreksi Outlier**")
        col_c.dataframe(df_prophet_clean.rename(columns={"ds":"Periode","y":"Jumlah"}), use_container_width=True)

    # ── Hitung periods ahead ───────────────────────────────────────────────
    last_date  = df_prophet_clean["ds"].max()
    target_dt  = pd.to_datetime(target_date)
    delta_days = (target_dt - last_date).days

    if   freq == "D":     periods_ahead = max(1, delta_days)
    elif freq == "W-MON": periods_ahead = max(1, delta_days // 7)
    else:                 periods_ahead = max(1, delta_days // 28)

    # ── Auto-tuning atau manual ────────────────────────────────────────────
    if manual_mode:
        best_params = _build_prophet_params(manual_cps, manual_sps, freq)
        with st.spinner("Melatih model dengan parameter manual…"):
            model, forecast = _train_final_model(df_prophet_clean, best_params, periods_ahead, freq)
        st.info(f"ℹ️ Mode Manual — CPS={manual_cps}, SPS={manual_sps}, Frekuensi={freq_label}")
    else:
        st.markdown("---")
        tuned_params = section_evaluasi_akurasi_optimal(
            df_prophet_raw, df_prophet_clean, freq, n_outlier, out_lower, out_upper
        )
        if tuned_params is None:
            tuned_params = _build_prophet_params(0.05, 5.0, freq)
        best_params = tuned_params
        with st.spinner("Melatih model final dengan parameter terbaik…"):
            model, forecast = _train_final_model(df_prophet_clean, best_params, periods_ahead, freq)

    # ── Grafik prediksi ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📈 Grafik Peramalan: **{pilihan_item}**")

    forecast_hist   = forecast[forecast["ds"] <= last_date]
    forecast_future = forecast[forecast["ds"] >  last_date]

    fig = go.Figure()
    # Rentang in-sample
    fig.add_trace(go.Scatter(
        x=forecast_hist["ds"].tolist()+forecast_hist["ds"].tolist()[::-1],
        y=forecast_hist["yhat_upper"].tolist()+forecast_hist["yhat_lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(37,99,235,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="Rentang In-Sample", hoverinfo="skip"
    ))
    # Data asli
    fig.add_trace(go.Scatter(
        x=df_prophet_raw["ds"], y=df_prophet_raw["y"],
        mode="lines+markers", name="Data Aktual (Asli)",
        line=dict(color="#94a3b8",width=1.5,dash="dot"), marker=dict(size=5,color="#94a3b8")
    ))
    # Data terkoreksi
    fig.add_trace(go.Scatter(
        x=df_prophet_clean["ds"], y=df_prophet_clean["y"],
        mode="lines+markers", name="Data Aktual (Terkoreksi)",
        line=dict(color="#2563eb",width=2), marker=dict(size=6,color="#2563eb")
    ))
    # Fit in-sample
    fig.add_trace(go.Scatter(
        x=forecast_hist["ds"], y=forecast_hist["yhat"],
        mode="lines", name="Fit Model (Historis)",
        line=dict(color="#7c3aed",width=1.5,dash="dash")
    ))
    # Prediksi masa depan
    if not forecast_future.empty:
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"].tolist()+forecast_future["ds"].tolist()[::-1],
            y=forecast_future["yhat_upper"].tolist()+forecast_future["yhat_lower"].tolist()[::-1],
            fill="toself", fillcolor="rgba(239,68,68,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="Rentang Toleransi", hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_future["ds"], y=forecast_future["yhat"].clip(lower=0),
            mode="lines+markers", name="Prediksi Masa Depan",
            line=dict(color="#ef4444",width=3,dash="dash"), marker=dict(size=7,color="#ef4444")
        ))

    fig.update_layout(
        xaxis_title="Periode Waktu", yaxis_title="Jumlah Kunjungan/Pasien",
        hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Dekomposisi tren ───────────────────────────────────────────────────
    with st.expander("🔍 Dekomposisi Tren & Musiman", expanded=False):
        st.caption("Memperlihatkan kontribusi tren jangka panjang, pola tahunan, dan efek hari libur.")
        fig_dec = go.Figure()
        fig_dec.add_trace(go.Scatter(x=forecast["ds"], y=forecast["trend"],
                                      mode="lines", name="Tren",
                                      line=dict(color="#2563eb",width=2)))
        if "yearly" in forecast.columns:
            fig_dec.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yearly"],
                                          mode="lines", name="Pola Tahunan",
                                          line=dict(color="#f59e0b",width=2,dash="dash")))
        if "holidays" in forecast.columns:
            fig_dec.add_trace(go.Scatter(x=forecast["ds"], y=forecast["holidays"],
                                          mode="lines", name="Efek Hari Libur",
                                          line=dict(color="#ef4444",width=2,dash="dot")))
        fig_dec.update_layout(title="Dekomposisi Komponen Model",
                               hovermode="x unified", margin=dict(t=40,b=20))
        st.plotly_chart(fig_dec, use_container_width=True)

    # ── Download ───────────────────────────────────────────────────────────
    if not forecast_future.empty:
        df_dl = forecast_future[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        df_dl.columns = ["Tanggal","Prediksi","Batas_Bawah","Batas_Atas"]
        df_dl[["Prediksi","Batas_Bawah","Batas_Atas"]] = (
            df_dl[["Prediksi","Batas_Bawah","Batas_Atas"]].clip(lower=0).round(0).astype(int)
        )
        st.download_button("📥 Download Data Prediksi (Excel)",
                            convert_df_to_excel(df_dl), f"prediksi_{pilihan_item}.xlsx")

    # ── Kesimpulan estimasi ────────────────────────────────────────────────
    st.markdown(f"### 📢 Kesimpulan Estimasi Hingga {pd.to_datetime(target_date).strftime('%d %B %Y')}")
    if not forecast_future.empty:
        total_estimasi      = int(round(forecast_future["yhat"].clip(lower=0).sum()))
        target_week         = forecast_future.iloc[-1]
        tgl_target_akhir    = target_week["ds"].strftime("%d %B %Y")
        est_kunjungan_akhir = max(0, int(round(target_week["yhat"])))
        batas_bawah_akhir   = max(0, int(round(target_week["yhat_lower"])))
        batas_atas_akhir    = max(0, int(round(target_week["yhat_upper"])))

        col_alert, col_m1, col_m2 = st.columns([2,1,1])
        with col_alert:
            st.markdown(f"""
                <div style="padding:1rem;border-radius:0.5rem;
                            background:rgba(59,130,246,0.1);
                            border:1px solid rgba(59,130,246,0.3);margin-bottom:1rem;">
                    <div class="highlight-estimasi">
                        📆 Hingga <b>{tgl_target_akhir}</b>, model memperkirakan total
                        <b>{total_estimasi} kunjungan/kasus</b> untuk <b>{pilihan_item}</b>.<br><br>
                        Pada periode terakhir diprediksi <b>{est_kunjungan_akhir} kunjungan</b>
                        (rentang: {batas_bawah_akhir}–{batas_atas_akhir}).
                    </div>
                </div>
            """, unsafe_allow_html=True)
        col_m1.metric("Total Akumulasi",        f"{total_estimasi} Pasien")
        col_m2.metric("Estimasi Periode Terakhir", f"{est_kunjungan_akhir} Pasien",
                       help=f"Batas Bawah: {batas_bawah_akhir} | Batas Atas: {batas_atas_akhir}")
    else:
        st.info("Tanggal prediksi terlalu dekat dengan data terakhir.")

    # ── Evaluasi akurasi mode manual ───────────────────────────────────────
    if manual_mode:
        st.markdown("---")
        with st.expander("📐 Evaluasi Akurasi Model Manual", expanded=False):
            total_days_cv = (df_prophet_clean["ds"].max() - df_prophet_clean["ds"].min()).days
            if total_days_cv < 60:
                st.warning("⚠️ Data terlalu pendek untuk cross-validation.")
            else:
                initial_days = max(30, int(total_days_cv*0.5))
                period_days  = max(14, int(total_days_cv*0.15))
                horizon_days = max(14, int(total_days_cv*0.25))
                try:
                    with st.spinner("Menjalankan cross-validation…"):
                        df_pm = _run_single_cv(best_params, df_prophet_clean,
                                               f"{initial_days} days",
                                               f"{period_days} days",
                                               f"{horizon_days} days")
                    mape = df_pm["mape"].mean()*100
                    mae  = df_pm["mae"].mean()
                    rmse = df_pm["rmse"].mean()
                    cov  = df_pm["coverage"].mean()*100
                    st.markdown(_render_akurasi_badge(mape), unsafe_allow_html=True)
                    st.markdown("")
                    mc1,mc2,mc3,mc4 = st.columns(4)
                    mc1.metric("MAE",      f"{mae:.1f}")
                    mc2.metric("RMSE",     f"{rmse:.1f}")
                    mc3.metric("MAPE",     f"{mape:.1f}%")
                    mc4.metric("Coverage", f"{cov:.1f}%")
                except Exception as e:
                    st.error(f"❌ Gagal CV: {e}")


# ========= HALAMAN LAIN =========
def page_data(df_filtered, filter_info):
    st.subheader("📄 Data & Unduhan")
    if df_filtered is None: return
    st.dataframe(df_filtered)
    st.download_button("💾 Download CSV",
                        df_filtered.to_csv(index=False).encode("utf-8"),
                        "data_puskesmas.csv","text/csv")

def page_quality(df):
    st.subheader("🧹 Kualitas Data")
    if df is None: return
    st.write(f"Duplikasi: {df.duplicated().sum()} baris")
    st.write("Missing Values:")
    st.dataframe(df.isna().sum().to_frame("Missing Count"))

def page_ai_assistant(df_filtered, filter_info, is_genai_configured):
    st.subheader("🤖 Agent AI Cerdas")
    if df_filtered is None or df_filtered.empty:
        st.warning("⚠️ Data belum ada. Silakan upload data terlebih dahulu."); return
    if not is_genai_configured:
        st.error("❌ API Key belum diset."); return

    total_data   = len(df_filtered)
    top_diagnosa = "-"
    if "diagnosa" in df_filtered.columns:
        s = df_filtered["diagnosa"].value_counts().head(5)
        top_diagnosa = ", ".join([f"{k} ({v} kasus)" for k,v in s.items()])
    top_poli = "-"
    if "poli" in df_filtered.columns:
        s = df_filtered["poli"].value_counts().head(3)
        top_poli = ", ".join([f"{k} ({v} kunjungan)" for k,v in s.items()])
    gender = (df_filtered["jenis_kelamin"].mode()[0]
              if "jenis_kelamin" in df_filtered.columns and not df_filtered["jenis_kelamin"].dropna().empty else "-")
    umur   = (df_filtered["kelompok_umur"].mode()[0]
              if "kelompok_umur"  in df_filtered.columns and not df_filtered["kelompok_umur"].dropna().empty  else "-")
    desa   = "-"
    if "desa" in df_filtered.columns:
        s = df_filtered["desa"].value_counts().head(3)
        desa = ", ".join([f"{k} ({v} pasien)" for k,v in s.items()])

    context_summary = f"""[STATISTIK DATA PASIEN SAAT INI]
- Total Pasien/Kunjungan: {total_data}
- 5 Penyakit Terbanyak: {top_diagnosa}
- 3 Poli Terpadat: {top_poli}
- Demografi Mayoritas: Jenis Kelamin {gender}, Kelompok Usia {umur}
- 3 Desa Asal Pasien Terbanyak: {desa}"""

    with st.expander("👁️ Konteks Data yang Dikirim ke AI", expanded=False):
        st.code(context_summary, language="markdown")
    st.info('💡 **Tips:** Coba tanyakan: *"Berdasarkan data penyakit terbanyak, program promkes apa yang paling mendesak?"*')
    user_q = st.text_area("Tanyakan strategi/analisis kesehatan:", placeholder="Ketik pertanyaan Anda di sini…")

    if st.button("Kirim Pertanyaan"):
        if not user_q.strip():
            st.warning("Pertanyaan tidak boleh kosong."); return
        prompt = f"""Anda adalah Analis Data dan Ahli Kesehatan Masyarakat profesional di UPT Puskesmas Purwosari (Bojonegoro).
Berikut ringkasan data pasien real-time:

{context_summary}

Jawablah pertanyaan pengguna secara spesifik, berbasis data di atas, praktis, dan terstruktur.

Pertanyaan: {user_q}"""
        with st.spinner("🤖 AI sedang menganalisis…"):
            try:
                model_ai = genai.GenerativeModel("gemini-2.5-flash")
                response = model_ai.generate_content(prompt)
                st.markdown("### 📊 Analisis AI:")
                st.markdown(f"""
                <div style="padding:1.5rem;border-radius:0.5rem;
                            background:var(--secondary-background-color);
                            border:1px solid rgba(148,163,184,0.4);">
                    {response.text}
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    st.warning("⏳ API sedang sibuk. Tunggu ~1 menit lalu coba lagi.")
                else:
                    st.error(f"❌ Gagal: {err}")

def page_cetak_laporan(df_filtered, filter_info):
    st.subheader("🖨️ Cetak Laporan PDF")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong."); return
    st.info("💡 Proses pembuatan PDF memerlukan beberapa detik untuk merender grafik.")

    if st.button("📄 Buat Dokumen PDF"):
        with st.spinner("Menyusun laporan…"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial","B",16)
                pdf.cell(0,8,"LAPORAN RINGKASAN KUNJUNGAN PASIEN",ln=True,align="C")
                pdf.set_font("Arial","B",14)
                pdf.cell(0,8,"UPT PUSKESMAS PURWOSARI - KAB. BOJONEGORO",ln=True,align="C")
                pdf.set_font("Arial","",10)
                pdf.cell(0,6,f"Waktu Cetak: {datetime.now().strftime('%d %B %Y %H:%M')}",ln=True,align="C")
                pdf.line(10,35,200,35); pdf.ln(10)

                pdf.set_font("Arial","B",12)
                pdf.cell(0,8,"1. Parameter Filter Aktif",ln=True)
                pdf.set_font("Arial","",11)
                if filter_info and any(filter_info.values()):
                    for k,v in filter_info.items():
                        if v: pdf.cell(0,6,f"- {k.replace('_',' ').title()}: {', '.join(map(str,v))}",ln=True)
                else:
                    pdf.cell(0,6,"- Menampilkan Semua Data",ln=True)
                pdf.ln(5)

                pdf.set_font("Arial","B",12)
                pdf.cell(0,8,"2. Ringkasan Kunjungan",ln=True)
                pdf.set_font("Arial","",11)
                pdf.cell(0,6,f"Total Kunjungan: {len(df_filtered)} kunjungan",ln=True)
                if "no_rm" in df_filtered.columns:
                    pdf.cell(0,6,f"Pasien Unik (RM): {df_filtered['no_rm'].nunique()} pasien",ln=True)
                pdf.ln(5)

                if "diagnosa" in df_filtered.columns:
                    pdf.set_font("Arial","B",12)
                    pdf.cell(0,8,"3. Top 5 Diagnosa Penyakit",ln=True)
                    pdf.set_font("Arial","",11)
                    for i,(p,j) in enumerate(df_filtered["diagnosa"].value_counts().head(5).items(),1):
                        pdf.cell(0,6,f"{i}. {p} ({j} kasus)",ln=True)
                    pdf.ln(5)

                if "jenis_kelamin" in df_filtered.columns:
                    pdf.set_font("Arial","B",12)
                    pdf.cell(0,8,"4. Demografi Jenis Kelamin",ln=True)
                    pdf.set_font("Arial","",11)
                    for jk,jml in df_filtered["jenis_kelamin"].value_counts().items():
                        pdf.cell(0,6,f"- {jk}: {jml} pasien",ln=True)

                pdf.add_page()
                pdf.set_font("Arial","B",14)
                pdf.cell(0,10,"LAMPIRAN VISUALISASI DATA",ln=True,align="C")
                pdf.line(10,20,200,20); pdf.ln(5)

                temp_images = []
                if "diagnosa" in df_filtered.columns:
                    top10 = df_filtered["diagnosa"].value_counts().head(10).reset_index()
                    top10.columns = ["Diagnosa","Jumlah"]
                    fig_d = px.bar(top10,x="Diagnosa",y="Jumlah",title="Top 10 Diagnosa",text_auto=True)
                    fd,path_d = tempfile.mkstemp(suffix=".png"); os.close(fd)
                    fig_d.write_image(path_d, engine="kaleido", width=800, height=500)
                    temp_images.append(path_d)
                    pdf.set_font("Arial","B",12)
                    pdf.cell(0,8,"A. Grafik Distribusi Penyakit",ln=True)
                    pdf.image(path_d,x=15,w=180); pdf.ln(5)

                if "tanggal_kunjungan" in df_filtered.columns:
                    trend = (df_filtered.groupby(["tahun","bulan","nama_bulan"])
                             .size().reset_index(name="count").sort_values(["tahun","bulan"]))
                    if not trend.empty:
                        trend["Periode"] = trend["nama_bulan"].astype(str)+" "+trend["tahun"].astype(str)
                        fig_t = px.line(trend,x="Periode",y="count",
                                        title="Tren Kunjungan Bulanan",markers=True)
                        fd,path_t = tempfile.mkstemp(suffix=".png"); os.close(fd)
                        fig_t.write_image(path_t, engine="kaleido", width=800, height=400)
                        temp_images.append(path_t)
                        pdf.set_font("Arial","B",12)
                        pdf.cell(0,8,"B. Tren Kunjungan Bulanan",ln=True)
                        pdf.image(path_t,x=15,w=180); pdf.ln(5)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    pdf.output(tmp_pdf.name)
                    with open(tmp_pdf.name,"rb") as f:
                        pdf_bytes = f.read()

                for p in temp_images:
                    if os.path.exists(p): os.remove(p)

                st.success("✅ Laporan PDF berhasil dibuat!")
                st.download_button(
                    "📥 Download Laporan PDF",
                    data=pdf_bytes,
                    file_name=f"Laporan_Puskesmas_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ========= MAIN =========
def main():
    df_filtered, filter_info = apply_filters(None)
    is_genai_configured      = get_gemini_client()

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigasi", [
        "Ringkasan Umum","Analisis Kunjungan","Analisis Penyakit",
        "Peta Persebaran","Analisis Pembiayaan","Data & Unduhan",
        "Kualitas Data","Prediksi ML","Agent AI","Cetak Laporan PDF"
    ])

    if   page == "Ringkasan Umum":      page_overview(df_filtered, filter_info)
    elif page == "Analisis Kunjungan":  page_kunjungan(df_filtered, filter_info)
    elif page == "Analisis Penyakit":   page_penyakit(df_filtered, filter_info)
    elif page == "Peta Persebaran":     page_peta_persebaran(df_filtered, filter_info)
    elif page == "Analisis Pembiayaan": page_pembiayaan(df_filtered, filter_info)
    elif page == "Data & Unduhan":      page_data(df_filtered, filter_info)
    elif page == "Kualitas Data":       page_quality(df_filtered)
    elif page == "Prediksi ML":         page_ml(df_filtered, filter_info)
    elif page == "Agent AI":            page_ai_assistant(df_filtered, filter_info, is_genai_configured)
    elif page == "Cetak Laporan PDF":   page_cetak_laporan(df_filtered, filter_info)

if __name__ == "__main__":
    main()
