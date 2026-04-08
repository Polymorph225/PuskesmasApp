import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime
import tempfile

# Library Machine Learning & AI
import google.generativeai as genai
from prophet import Prophet

# ========== INISIALISASI SESSION STATE (MEMORI APLIKASI) ==========
if "ml_fig" not in st.session_state:
    st.session_state["ml_fig"] = None
if "ml_title" not in st.session_state:
    st.session_state["ml_title"] = None
if "ai_response" not in st.session_state:
    st.session_state["ai_response"] = None

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
        .block-container { padding: 1.5rem 2rem 3rem 2rem; }
        h1 { font-weight: 700 !important; }
        div[data-testid="metric-container"] {
            padding: 0.75rem 1rem; border-radius: 0.75rem;
            background-color: var(--secondary-background-color); 
            border: 1px solid var(--faded-text-color);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
        }
        hr { margin: 0.75rem 0 1rem 0; }
        .highlight-estimasi { color: #1d4ed8 !important; line-height: 1.5; font-size: 1.05rem; font-weight: 800; }
        </style>
        """,
        unsafe_allow_html=True,
    )
inject_custom_css()

# ========== HEADER UTAMA ==========
st.title("📊 Dashboard Analisis Data Puskesmas")
st.markdown("Dashboard cerdas terintegrasi Machine Learning & AI untuk tata kelola data UPT Puskesmas Purwosari.")

# ========= FUNGSI AI & DATA CLEANING (TIDAK BERUBAH) =========
@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st.secrets, "get") else None
    if not api_key: api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ API key Gemini belum diset.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True 
    except: return False

TARGET_COLS = ["tanggal_kunjungan", "no_rm", "umur", "jenis_kelamin", "poli", "diagnosa", "pembiayaan", "desa"]

def _normalize_col_name(col): return re.sub(r"[^a-z0-9]", "", str(col).strip().lower())

def _build_column_mapping(df_raw):
    mapping = {}
    alias_dict = {
        "tanggal_kunjungan": ["tanggalkunjungan", "tglkunjungan", "tanggal", "tgl", "visitdate"],
        "no_rm": ["norm", "norekammedis", "rekammedis"],
        "umur": ["umur", "usia", "age"],
        "jenis_kelamin": ["jeniskelamin", "jk", "sex", "kelamin"],
        "poli": ["poli", "politujuan", "unit", "poliklinik"],
        "diagnosa": ["diagnosa", "dxutama", "icd10", "diagnosis"],
        "pembiayaan": ["pembiayaan", "carabayar", "penjamin", "jaminan"],
        "desa": ["desa", "alamatdesa", "kelurahan", "namadesa"],
    }
    norm_cols = {col: _normalize_col_name(col) for col in df_raw.columns}
    for std_name, alias_list in alias_dict.items():
        for raw_col, norm_key in norm_cols.items():
            if norm_key in alias_list:
                mapping[raw_col] = std_name
                break
    return mapping

def clean_raw_data(df_raw):
    if df_raw is None or df_raw.empty: return df_raw
    df = df_raw.rename(columns=_build_column_mapping(df_raw)).copy()
    keep_cols = [c for c in TARGET_COLS if c in df.columns]
    df = df[keep_cols].copy()
    if "tanggal_kunjungan" in df.columns: df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
    if "umur" in df.columns: df["umur"] = pd.to_numeric(df["umur"].astype(str).str.extract(r'(\d+)')[0], errors="coerce")
    if "jenis_kelamin" in df.columns: df["jenis_kelamin"] = df["jenis_kelamin"].astype(str).str.lower().map({"l":"Laki-Laki", "lk":"Laki-Laki", "laki-laki":"Laki-Laki", "1":"Laki-Laki", "p":"Perempuan", "pr":"Perempuan", "perempuan":"Perempuan", "2":"Perempuan"}).fillna("Tidak Diketahui")
    for col in ["poli", "diagnosa", "pembiayaan", "desa", "no_rm"]:
        if col in df.columns: df[col] = df[col].astype(str).str.strip().str.title().replace({"Nan": np.nan, "None": np.nan})
    return df

@st.cache_data
def load_data(file):
    ext = file.name.lower().split(".")[-1]
    df_raw = pd.read_csv(file) if ext == "csv" else pd.read_excel(file, engine="openpyxl" if ext == "xlsx" else "xlrd")
    return clean_raw_data(df_raw), df_raw

def preprocess_data(df):
    if df is None or df.empty: return df
    if "umur" in df.columns:
        df["kelompok_umur"] = pd.cut(df["umur"], bins=[0, 1, 5, 15, 25, 45, 60, 200], labels=["<1 th", "1-4 th", "5-14 th", "15-24 th", "25-44 th", "45-59 th", "60+ th"], right=False)
    if "tanggal_kunjungan" in df.columns:
        df["tahun"] = df["tanggal_kunjungan"].dt.year
        df["bulan"] = df["tanggal_kunjungan"].dt.month
        df["nama_bulan"] = df["tanggal_kunjungan"].dt.strftime("%b")
    return df

def apply_filters(_):
    with st.sidebar:
        st.markdown("## 🏥 Data Puskesmas")
        uploaded_file = st.file_uploader("Upload data (CSV/Excel)", type=["csv", "xlsx", "xls"])
        if not uploaded_file: return None, None
        df_clean, _ = load_data(uploaded_file)
        df = preprocess_data(df_clean)
        
        date_range = tahun_p = poli_p = jk_p = bayar_p = umur_p = desa_p = None
        st.markdown("### 🔍 Filter")
        with st.expander("Filter Waktu & Poli", expanded=True):
            if "tanggal_kunjungan" in df.columns: date_range = st.date_input("Rentang", value=[df["tanggal_kunjungan"].min().date(), df["tanggal_kunjungan"].max().date()])
            if "poli" in df.columns: poli_p = st.multiselect("Poli", options=sorted(df["poli"].dropna().unique()))
        with st.expander("Filter Lainnya", expanded=False):
            if "jenis_kelamin" in df.columns: jk_p = st.multiselect("Gender", options=sorted(df["jenis_kelamin"].dropna().unique()))
            if "kelompok_umur" in df.columns: umur_p = st.multiselect("Umur", options=df["kelompok_umur"].dropna().unique())
            if "desa" in df.columns: desa_p = st.multiselect("Desa", options=sorted(df["desa"].dropna().unique()))
            if "pembiayaan" in df.columns: bayar_p = st.multiselect("Pembiayaan", options=sorted(df["pembiayaan"].dropna().unique()))

    df_filtered = df.copy()
    if date_range and len(date_range) == 2: df_filtered = df_filtered[(df_filtered["tanggal_kunjungan"].dt.date >= date_range[0]) & (df_filtered["tanggal_kunjungan"].dt.date <= date_range[1])]
    if poli_p: df_filtered = df_filtered[df_filtered["poli"].isin(poli_p)]
    if jk_p: df_filtered = df_filtered[df_filtered["jenis_kelamin"].isin(jk_p)]
    if bayar_p: df_filtered = df_filtered[df_filtered["pembiayaan"].isin(bayar_p)]
    if umur_p: df_filtered = df_filtered[df_filtered["kelompok_umur"].isin(umur_p)]
    if desa_p: df_filtered = df_filtered[df_filtered["desa"].isin(desa_p)]
    
    return df_filtered, {"poli": poli_p, "jenis_kelamin": jk_p, "pembiayaan": bayar_p, "kelompok_umur": umur_p, "desa": desa_p}

def show_active_filters(filter_info):
    if not filter_info: return
    chips = [f"{k.title()}: {', '.join(map(str, v))}" for k, v in filter_info.items() if v]
    if chips: st.caption("🎯 **Filter aktif:** " + " | ".join(chips))

# ========= HALAMAN VISUALISASI =========
def page_overview(df, filter_info):
    st.subheader("📌 Ringkasan Umum")
    show_active_filters(filter_info)
    if df is None or df.empty: return st.warning("Data kosong.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Kunjungan", len(df))
    if "no_rm" in df.columns: c2.metric("Pasien Unik", df["no_rm"].nunique())
    if "poli" in df.columns: c3.metric("Poli Aktif", df["poli"].nunique())
    if "diagnosa" in df.columns: c4.metric("Diagnosa", df["diagnosa"].nunique())
    
    st.markdown("---")
    if "tanggal_kunjungan" in df.columns:
        trend = df.groupby(["tahun", "bulan", "nama_bulan"]).size().reset_index(name="count").sort_values(["tahun", "bulan"])
        trend["label"] = trend["nama_bulan"] + "-" + trend["tahun"].astype(str)
        fig = px.line(trend, x="label", y="count", title="Tren Kunjungan", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    if "poli" in df.columns:
        fig_poli = px.bar(df["poli"].value_counts().reset_index(), x='poli', y='count', title="Distribusi Poli")
        st.plotly_chart(fig_poli, use_container_width=True)

def page_kunjungan(df, filter_info):
    st.subheader("👥 Analisis Kunjungan")
    if df is None or df.empty: return
    c1, c2 = st.columns(2)
    if "jenis_kelamin" in df.columns:
        fig_jk = px.pie(df, names='jenis_kelamin', title="Jenis Kelamin")
        c1.plotly_chart(fig_jk, use_container_width=True)
    if "kelompok_umur" in df.columns:
        fig_umur = px.bar(df["kelompok_umur"].value_counts().sort_index().reset_index(), x='kelompok_umur', y='count', title="Kelompok Umur")
        c2.plotly_chart(fig_umur, use_container_width=True)

def page_penyakit(df, filter_info):
    st.subheader("🦠 Analisis Penyakit")
    if df is None or df.empty or "diagnosa" not in df.columns: return
    top_n = st.slider("Jumlah", 5, 30, 10)
    fig_diag = px.bar(df["diagnosa"].value_counts().head(top_n).reset_index(), x='diagnosa', y='count', title=f"Top {top_n} Diagnosa", text_auto=True)
    st.plotly_chart(fig_diag, use_container_width=True)

def page_peta(df, filter_info):
    st.subheader("🗺️ Persebaran & Analisis Desa")
    if df is None or df.empty or "desa" not in df.columns: return
    
    col_p, _ = st.columns([3, 1])
    pilihan_penyakit = col_p.selectbox("Diagnosa:", ["-- Semua --"] + df["diagnosa"].value_counts().head(20).index.tolist()) if "diagnosa" in df.columns else "-- Semua --"
    
    df_map = df[df["diagnosa"] == pilihan_penyakit].copy() if pilihan_penyakit != "-- Semua --" else df.copy()
    df_grouped = df_map.groupby(["desa"]).size().reset_index(name="jumlah_kasus").sort_values("jumlah_kasus", ascending=False)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Top Desa Terdampak")
        fig_desa = px.bar(df_grouped.head(10), x='desa', y='jumlah_kasus', text_auto=True)
        st.plotly_chart(fig_desa, use_container_width=True)
    with c2:
        st.markdown("#### Detail Kasus per Desa")
        st.dataframe(df_grouped, use_container_width=True, hide_index=True)

def page_pembiayaan(df, filter_info):
    st.subheader("💳 Analisis Pembiayaan")
    if df is None or df.empty or "pembiayaan" not in df.columns: return
    fig_bayar = px.pie(df, names='pembiayaan', title="Proporsi Pembiayaan", hole=0.4)
    st.plotly_chart(fig_bayar, use_container_width=True)

def page_ml(df, filter_info):
    st.subheader("📈 Prediksi Prophet")
    if df is None or df.empty or "tanggal_kunjungan" not in df.columns or "diagnosa" not in df.columns: return
    
    df_ml = df.dropna(subset=["tanggal_kunjungan"]).copy()
    top_items = df_ml["diagnosa"].value_counts().head(20).index.tolist()
    
    c1, c2 = st.columns(2)
    pilihan_item = c1.selectbox("Pilih Penyakit:", top_items)
    target_date = c2.date_input("Prediksi Sampai:", value=df_ml["tanggal_kunjungan"].max().date() + pd.Timedelta(days=30))
    
    df_item = df_ml[df_ml["diagnosa"] == pilihan_item].copy()
    if len(df_item) < 10: return st.error("Data terlalu sedikit.")

    weekly = df_item.groupby(pd.Grouper(key="tanggal_kunjungan", freq="W-MON")).size().reset_index(name="y")
    weekly.rename(columns={"tanggal_kunjungan": "ds"}, inplace=True)
    
    with st.spinner("AI memproses..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(weekly)
        future = m.make_future_dataframe(periods=max(1, (target_date - weekly["ds"].max().date()).days // 7), freq='W-MON')
        forecast = m.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly['ds'], y=weekly['y'], mode='lines+markers', name='Aktual', line=dict(color='blue')))
    forecast_fut = forecast[forecast['ds'] > pd.to_datetime(weekly["ds"].max())]
    fig.add_trace(go.Scatter(x=forecast_fut['ds'], y=forecast_fut['yhat'], mode='lines+markers', name='Prediksi', line=dict(color='red', dash='dash')))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # SIMPAN KE SESSION STATE UNTUK DI CETAK PDF NANTI
    st.session_state["ml_fig"] = fig
    st.session_state["ml_title"] = f"Prediksi Tren Penyakit {pilihan_item} s/d {target_date.strftime('%d %b %Y')}"
    
    st.success("Visualisasi telah disimpan di memori untuk keperluan cetak laporan PDF.")

def page_ai(df, is_genai_configured):
    st.subheader("🤖 Agent AI Cerdas")
    if not is_genai_configured: return st.error("API Key belum diset.")
    if df is None or df.empty: return
    
    user_q = st.text_area("Tanyakan strategi/analisis:", placeholder="Contoh: Apa program promkes yang cocok berdasarkan data?")
    if st.button("Kirim ke AI") and user_q:
        summary = f"Total Kunjungan: {len(df)}. Top Penyakit: {df['diagnosa'].mode()[0] if 'diagnosa' in df.columns else '-'}"
        prompt = f"Anda Ahli Kesmas Puskesmas Purwosari. Data saat ini: {summary}. Jawab: {user_q}"
        
        with st.spinner("Menganalisis..."):
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                res = model.generate_content(prompt)
                st.session_state["ai_response"] = res.text # Simpan ke memori
            except Exception as e: st.error(f"Error: {e}")

    # Tampilkan jika ada memori
    if st.session_state["ai_response"]:
        st.markdown("### 📊 Rekomendasi AI Terakhir:")
        st.info(st.session_state["ai_response"])

# ================== FUNGSI GENERATOR PDF KOMPREHENSIF ==================
def clean_text_for_pdf(text):
    """Menghapus Markdown dan karakter khusus agar FPDF 1.7.2 tidak crash"""
    if not text: return ""
    text = text.replace("**", "").replace("*", "").replace("#", "")
    return text.encode('latin-1', 'ignore').decode('latin-1')

def add_plotly_to_pdf(pdf, fig, title, temp_list, width=180, height=400):
    """Fungsi pembantu mengekspor plot ke gambar lalu ke PDF"""
    if fig is None: return
    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        fig.write_image(path, engine="kaleido", width=800, height=height)
        temp_list.append(path)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, clean_text_for_pdf(title), ln=True)
        pdf.image(path, x=(210-width)/2, w=width) # Center image
        pdf.ln(5)
    except Exception as e:
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 8, f"[Gagal merender grafik: {str(e)}]", ln=True)

def page_cetak_laporan(df, filter_info):
    st.subheader("🖨️ Cetak Laporan Eksekutif (Lengkap Data & Grafik)")
    if df is None or df.empty: return st.warning("Data kosong.")

    st.write("Sistem akan menggabungkan Analisis Kunjungan, Pembiayaan, Pemetaan Desa, hasil prediksi Machine Learning terakhir, dan Rekomendasi AI ke dalam satu dokumen PDF resmi.")

    if st.button("📄 Generate Laporan PDF"):
        with st.spinner("Memproses grafik, ML, dan AI menjadi dokumen PDF..."):
            pdf = FPDF()
            temp_images = []
            
            # --- HALAMAN 1: KOP & RINGKASAN TEKS ---
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 8, "LAPORAN ANALISIS KUNJUNGAN PASIEN", ln=True, align='C')
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 8, "UPT PUSKESMAS PURWOSARI - KAB. BOJONEGORO", ln=True, align='C')
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, f"Dicetak pada: {datetime.now().strftime('%d %b %Y %H:%M')}", ln=True, align='C')
            pdf.line(10, 35, 200, 35)
            pdf.ln(10)

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "1. Informasi Parameter Data", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.cell(0, 6, f"Total Data Tersedia: {len(df)} Kunjungan", ln=True)
            pdf.ln(5)

            # --- HALAMAN GRAFIK DASAR ---
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "A. ANALISIS KUNJUNGAN & DEMOGRAFI", ln=True)
            
            if "kelompok_umur" in df.columns:
                fig_umur = px.bar(df["kelompok_umur"].value_counts().sort_index().reset_index(), x='kelompok_umur', y='count', title="Distribusi Kelompok Umur")
                add_plotly_to_pdf(pdf, fig_umur, "1. Grafik Kelompok Umur", temp_images, height=350)
            
            if "jenis_kelamin" in df.columns:
                fig_jk = px.pie(df, names='jenis_kelamin', title="Proporsi Jenis Kelamin")
                add_plotly_to_pdf(pdf, fig_jk, "2. Grafik Jenis Kelamin", temp_images, height=350)

            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "B. ANALISIS PENYAKIT & PEMBIAYAAN", ln=True)
            
            if "diagnosa" in df.columns:
                fig_diag = px.bar(df["diagnosa"].value_counts().head(10).reset_index(), x='diagnosa', y='count', title="Top 10 Diagnosa Penyakit")
                add_plotly_to_pdf(pdf, fig_diag, "3. Top 10 Penyakit Terbanyak", temp_images, height=400)
                
            if "pembiayaan" in df.columns:
                fig_bayar = px.pie(df, names='pembiayaan', hole=0.4, title="Metode Pembiayaan Pasien")
                add_plotly_to_pdf(pdf, fig_bayar, "4. Distribusi Pembiayaan", temp_images, height=350)

            # --- HALAMAN DESA (GRAFIK & TABEL) ---
            if "desa" in df.columns:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "C. ANALISIS WILAYAH / DESA TERDAMPAK", ln=True)
                
                df_desa = df.groupby(["desa"]).size().reset_index(name="kasus").sort_values("kasus", ascending=False)
                fig_desa = px.bar(df_desa.head(10), x='desa', y='kasus', title="Top 10 Desa Terdampak")
                add_plotly_to_pdf(pdf, fig_desa, "5. Grafik Top Desa Terdampak", temp_images, height=350)

                # Tabel Desa
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 8, "Tabel Rincian Kasus per Desa:", ln=True)
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(120, 8, "Nama Desa", 1)
                pdf.cell(50, 8, "Jumlah Kasus", 1, ln=True, align='C')
                
                pdf.set_font("Arial", '', 10)
                for _, row in df_desa.iterrows():
                    pdf.cell(120, 7, clean_text_for_pdf(str(row['desa'])), 1)
                    pdf.cell(50, 7, str(row['kasus']), 1, ln=True, align='C')

            # --- HALAMAN PROPHET ML (DARI SESSION STATE) ---
            if st.session_state["ml_fig"] is not None:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "D. PREDIKSI & PERAMALAN TREN (MACHINE LEARNING)", ln=True)
                add_plotly_to_pdf(pdf, st.session_state["ml_fig"], st.session_state["ml_title"], temp_images, height=450)
                pdf.set_font("Arial", '', 10)
                pdf.multi_cell(0, 6, "Catatan: Prediksi di atas menggunakan algoritma Prophet berdasarkan pola deret waktu historis kunjungan penyakit yang dipilih pada menu Prediksi ML.")

            # --- HALAMAN AI (DARI SESSION STATE) ---
            if st.session_state["ai_response"] is not None:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "E. REKOMENDASI STRATEGIS (KECERDASAN BUATAN)", ln=True)
                pdf.ln(5)
                
                pdf.set_font("Arial", '', 11)
                teks_ai_bersih = clean_text_for_pdf(st.session_state["ai_response"])
                pdf.multi_cell(0, 6, teks_ai_bersih)

            # --- FINALISASI PDF ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pdf.output(tmp_pdf.name)
                with open(tmp_pdf.name, "rb") as f: pdf_bytes = f.read()

            for img_path in temp_images:
                if os.path.exists(img_path): os.remove(img_path)

            st.success("✅ Laporan LENGKAP berhasil di-generate!")
            st.download_button("📥 Download Laporan Eksekutif PDF", data=pdf_bytes, file_name=f"Laporan_Puskesmas_{datetime.now().strftime('%d%m%Y')}.pdf", mime="application/pdf", type="primary")

# ========= MAIN ROUTING =========
def main():
    df, filter_info = apply_filters(None)
    is_genai_ready = get_gemini_client()

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigasi", ["Ringkasan Umum", "Analisis Kunjungan", "Analisis Penyakit", "Peta Persebaran Desa", "Analisis Pembiayaan", "Prediksi Prophet ML", "Agent AI Rekomendasi", "Cetak Laporan PDF Terpadu"])
    
    if page == "Ringkasan Umum": page_overview(df, filter_info)
    elif page == "Analisis Kunjungan": page_kunjungan(df, filter_info)
    elif page == "Analisis Penyakit": page_penyakit(df, filter_info)
    elif page == "Peta Persebaran Desa": page_peta(df, filter_info)
    elif page == "Analisis Pembiayaan": page_pembiayaan(df, filter_info)
    elif page == "Prediksi Prophet ML": page_ml(df, filter_info)
    elif page == "Agent AI Rekomendasi": page_ai(df, is_genai_ready)
    elif page == "Cetak Laporan PDF Terpadu": page_cetak_laporan(df, filter_info)

if __name__ == "__main__":
    main()
