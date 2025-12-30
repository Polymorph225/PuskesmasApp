import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Dashboard Analisis Data Puskesmas",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dashboard Analisis Data Puskesmas")

st.write(
    """
    Aplikasi ini membantu menganalisa data kunjungan Puskesmas berdasarkan poli, diagnosa, 
    jenis kelamin, kelompok umur, pembiayaan, dan wilayah pelayanan.
    Silakan upload data kunjungan dalam format CSV/Excel melalui sidebar.
    """
)

# ========= FUNGSI UTAMA PERSIAPAN DATA =========

@st.cache_data
def load_data(file):
    """Membaca file CSV / Excel menjadi DataFrame."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def preprocess_data(df):
    """Membersihkan dan menambah kolom turunan."""
    # Normalisasi nama kolom (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Peta berbagai variasi nama kolom ke nama baku
    rename_map = {
        "tanggal kunjungan": "tanggal_kunjungan",
        "tgl kunjungan": "tanggal_kunjungan",
        "tgl_kunjungan": "tanggal_kunjungan",

        "no rm": "no_rm",
        "no_rm": "no_rm",
        "norm": "no_rm",

        "jenis kelamin": "jenis_kelamin",
        "jk": "jenis_kelamin",

        "alamat desa": "alamat_desa",
        "desa": "alamat_desa",

        "poli pelayanan": "poli",
        "poli_layanan": "poli",

        "cara bayar": "pembiayaan",
        "pembayaran": "pembiayaan",
    }

    df = df.rename(columns=rename_map)

    # Konversi tanggal kunjungan
    if "tanggal_kunjungan" in df.columns:
        df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
    
    # Pastikan umur numerik jika ada
    if "umur" in df.columns:
        df["umur"] = pd.to_numeric(df["umur"], errors="coerce")
    
    # Kelompok umur
    if "umur" in df.columns:
        bins = [0, 1, 5, 15, 25, 45, 60, 200]
        labels = ["<1 th", "1-4 th", "5-14 th", "15-24 th", "25-44 th", "45-59 th", "60+ th"]
        df["kelompok_umur"] = pd.cut(df["umur"], bins=bins, labels=labels, right=False)
    
    # Tambahan kolom tahun & bulan (angka & nama)
    if "tanggal_kunjungan" in df.columns:
        df["tahun"] = df["tanggal_kunjungan"].dt.year
        df["bulan"] = df["tanggal_kunjungan"].dt.month
        df["nama_bulan"] = df["tanggal_kunjungan"].dt.strftime("%b")
    
    # Standarisasi beberapa kolom teks jika ada
    for col in ["poli", "jenis_kelamin", "pembiayaan", "diagnosa", "alamat_desa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    
    return df

def apply_filters(_):
    """
    Buat & terapkan filter dari sidebar.
    Mengembalikan:
    - df_filtered: DataFrame terfilter
    - filter_info: dict info filter
    """
    with st.sidebar:
        st.header("📁 Upload & Filter Data")
        uploaded_file = st.file_uploader("Upload data kunjungan", type=["csv", "xlsx"])
        
        if uploaded_file is None:
            st.info("Silakan upload file CSV/Excel yang berisi data kunjungan.")
            return None, None
        
        df = load_data(uploaded_file)
        df = preprocess_data(df)
        
        st.success(f"Data berhasil dimuat. Total baris: {len(df)}")
        
        # ---------- Filter tanggal ----------
        if "tanggal_kunjungan" in df.columns:
            min_date = df["tanggal_kunjungan"].min()
            max_date = df["tanggal_kunjungan"].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = st.date_input(
                    "Rentang tanggal kunjungan",
                    value=[min_date.date(), max_date.date()],
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
            else:
                date_range = None
        else:
            date_range = None
        
        # ---------- Filter tahun ----------
        tahun_pilihan = None
        if "tahun" in df.columns:
            tahun_unik = sorted(df["tahun"].dropna().unique())
            if len(tahun_unik) > 0:
                tahun_pilihan = st.multiselect(
                    "Filter tahun",
                    options=tahun_unik,
                    default=tahun_unik
                )
        
        # ---------- Filter poli ----------
        poli_pilihan = None
        if "poli" in df.columns:
            poli_unik = sorted(df["poli"].dropna().unique())
            poli_pilihan = st.multiselect(
                "Filter poli",
                options=poli_unik,
                default=poli_unik
            )
        
        # ---------- Filter jenis kelamin ----------
        jk_pilihan = None
        if "jenis_kelamin" in df.columns:
            jk_unik = sorted(df["jenis_kelamin"].dropna().unique())
            jk_pilihan = st.multiselect(
                "Filter jenis kelamin",
                options=jk_unik,
                default=jk_unik
            )
        
        # ---------- Filter pembiayaan ----------
        bayar_pilihan = None
        if "pembiayaan" in df.columns:
            bayar_unik = sorted(df["pembiayaan"].dropna().unique())
            bayar_pilihan = st.multiselect(
                "Filter pembiayaan",
                options=bayar_unik,
                default=bayar_unik
            )
        
        # ---------- Filter kelompok umur ----------
        kelompok_umur_pilihan = None
        if "kelompok_umur" in df.columns:
            ku_unik = [x for x in df["kelompok_umur"].dropna().unique()]
            kelompok_umur_pilihan = st.multiselect(
                "Filter kelompok umur",
                options=ku_unik,
                default=ku_unik
            )
        
        # ---------- Filter desa/kelurahan ----------
        desa_pilihan = None
        if "alamat_desa" in df.columns:
            desa_unik = sorted(df["alamat_desa"].dropna().unique())
            desa_pilihan = st.multiselect(
                "Filter desa/kelurahan",
                options=desa_unik,
                default=desa_unik
            )
    
    # Terapkan filter di luar sidebar
    df_filtered = df.copy()
    
    if date_range is not None and len(date_range) == 2:
        start_date, end_date = date_range
        if "tanggal_kunjungan" in df_filtered.columns:
            mask = (
                (df_filtered["tanggal_kunjungan"].dt.date >= start_date)
                & (df_filtered["tanggal_kunjungan"].dt.date <= end_date)
            )
            df_filtered = df_filtered[mask]
    
    if tahun_pilihan is not None and len(tahun_pilihan) > 0:
        df_filtered = df_filtered[df_filtered["tahun"].isin(tahun_pilihan)]
    
    if poli_pilihan is not None and len(poli_pilihan) > 0:
        df_filtered = df_filtered[df_filtered["poli"].isin(poli_pilihan)]
    
    if jk_pilihan is not None and len(jk_pilihan) > 0:
        df_filtered = df_filtered[df_filtered["jenis_kelamin"].isin(jk_pilihan)]
    
    if bayar_pilihan is not None and len(bayar_pilihan) > 0:
        df_filtered = df_filtered[df_filtered["pembiayaan"].isin(bayar_pilihan)]
    
    if kelompok_umur_pilihan is not None and len(kelompok_umur_pilihan) > 0:
        df_filtered = df_filtered[df_filtered["kelompok_umur"].isin(kelompok_umur_pilihan)]
    
    if desa_pilihan is not None and len(desa_pilihan) > 0:
        df_filtered = df_filtered[df_filtered["alamat_desa"].isin(desa_pilihan)]
    
    filter_info = {
        "date_range": date_range,
        "tahun": tahun_pilihan,
        "poli": poli_pilihan,
        "jenis_kelamin": jk_pilihan,
        "pembiayaan": bayar_pilihan,
        "kelompok_umur": kelompok_umur_pilihan,
        "desa": desa_pilihan,
    }
    
    return df_filtered, filter_info

# ========= HALAMAN-HALAMAN =========

def page_overview(df_filtered, filter_info):
    st.subheader("📌 Ringkasan Umum")
    
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data untuk ditampilkan. Periksa kembali filter Anda.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_kunjungan = len(df_filtered)
    total_pasien = df_filtered["no_rm"].nunique() if "no_rm" in df_filtered.columns else np.nan
    total_poli = df_filtered["poli"].nunique() if "poli" in df_filtered.columns else np.nan
    total_diagnosa = df_filtered["diagnosa"].nunique() if "diagnosa" in df_filtered.columns else np.nan
    
    col1.metric("Total Kunjungan", int(total_kunjungan))
    if not np.isnan(total_pasien):
        col2.metric("Jumlah Pasien Unik", int(total_pasien))
    if not np.isnan(total_poli):
        col3.metric("Jumlah Poli Aktif", int(total_poli))
    if not np.isnan(total_diagnosa):
        col4.metric("Jumlah Diagnosa Berbeda", int(total_diagnosa))
    
    st.markdown("---")
    
    # Kunjungan per bulan
    if "tanggal_kunjungan" in df_filtered.columns:
        st.markdown("### Tren Kunjungan per Bulan")
        kunjungan_bulan = (
            df_filtered
            .groupby(["tahun", "bulan", "nama_bulan"])
            .size()
            .reset_index(name="jumlah_kunjungan")
            .sort_values(["tahun", "bulan"])
        )
        if len(kunjungan_bulan) > 0:
            kunjungan_bulan["label_bulan"] = (
                kunjungan_bulan["nama_bulan"].astype(str) 
                + "-" + kunjungan_bulan["tahun"].astype(str)
            )
            chart_data = kunjungan_bulan.set_index("label_bulan")["jumlah_kunjungan"]
            st.line_chart(chart_data)
        else:
            st.info("Data tren kunjungan per bulan tidak tersedia.")
    
    # Kunjungan per poli
    if "poli" in df_filtered.columns:
        st.markdown("### Distribusi Kunjungan per Poli")
        poli_counts = df_filtered["poli"].value_counts().sort_values(ascending=False)
        if len(poli_counts) > 0:
            st.bar_chart(poli_counts)
            st.dataframe(poli_counts.rename("Jumlah Kunjungan"))
    
    # Top diagnosa
    if "diagnosa" in df_filtered.columns:
        st.markdown("### Top 10 Diagnosa Terbanyak")
        top_diag = df_filtered["diagnosa"].value_counts().head(10)
        st.bar_chart(top_diag)
        st.dataframe(top_diag.rename("Jumlah Kunjungan"))

def page_kunjungan(df_filtered, filter_info):
    st.subheader("👥 Analisis Kunjungan & Pasien")
        st.write("Kolom yang terdeteksi:", df_filtered.columns.tolist())
    
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data untuk ditampilkan. Periksa kembali filter Anda.")
        return
    
    col1, col2 = st.columns(2)
    
    # Distribusi jenis kelamin
    if "jenis_kelamin" in df_filtered.columns:
        jk_counts = df_filtered["jenis_kelamin"].value_counts()
        col1.markdown("#### Distribusi Kunjungan per Jenis Kelamin")
        col1.bar_chart(jk_counts)
        col1.dataframe(jk_counts.rename("Jumlah Kunjungan"))
    
    # Distribusi kelompok umur
    if "kelompok_umur" in df_filtered.columns:
        umur_counts = (
            df_filtered["kelompok_umur"]
            .value_counts()
            .sort_index()
        )
        col2.markdown("#### Distribusi Kunjungan per Kelompok Umur")
        col2.bar_chart(umur_counts)
        col2.dataframe(umur_counts.rename("Jumlah Kunjungan"))
    
    st.markdown("---")
    
    # Tabel ringkasan per poli
    if "poli" in df_filtered.columns:
        st.markdown("### Ringkasan Kunjungan per Poli")
        agg_cols = {}
        if "no_rm" in df_filtered.columns:
            agg_cols["no_rm"] = "nunique"
        if "diagnosa" in df_filtered.columns:
            agg_cols["diagnosa"] = "nunique"
        
        ringkasan = (
            df_filtered
            .groupby("poli")
            .agg(
                jumlah_kunjungan=("poli", "size"),
                **{k: (k, v) for k, v in agg_cols.items()}
            )
            .sort_values("jumlah_kunjungan", ascending=False)
        )
        st.dataframe(ringkasan)
    
    # Crosstab poli x jenis kelamin
    if "poli" in df_filtered.columns and "jenis_kelamin" in df_filtered.columns:
        st.markdown("### Crosstab Poli x Jenis Kelamin")
        ctab = pd.crosstab(df_filtered["poli"], df_filtered["jenis_kelamin"])
        st.dataframe(ctab)

def page_penyakit(df_filtered, filter_info):
    st.subheader("🦠 Analisis Penyakit / Diagnosa")
    
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data untuk ditampilkan. Periksa kembali filter Anda.")
        return
    
    # Pilih poli (opsional) untuk fokus analisis
    if "poli" in df_filtered.columns:
        poli_opsi = ["(Semua Poli)"] + sorted(df_filtered["poli"].dropna().unique().tolist())
        poli_fokus = st.selectbox("Fokus pada poli tertentu", options=poli_opsi)
        if poli_fokus != "(Semua Poli)":
            df_filtered = df_filtered[df_filtered["poli"] == poli_fokus]
    
    # Top diagnosa
    if "diagnosa" in df_filtered.columns:
        top_n = st.slider("Tampilkan berapa diagnosa terbanyak", min_value=5, max_value=20, value=10)
        top_diag = df_filtered["diagnosa"].value_counts().head(top_n)
        
        st.markdown("### Diagnosa Terbanyak")
        st.bar_chart(top_diag)
        st.dataframe(top_diag.rename("Jumlah Kunjungan"))
    
    # Crosstab diagnosa x jenis kelamin
    if "diagnosa" in df_filtered.columns and "jenis_kelamin" in df_filtered.columns:
        st.markdown("### Crosstab Diagnosa x Jenis Kelamin (Top 10 Diagnosa)")
        top10_diag = df_filtered["diagnosa"].value_counts().head(10).index
        mask = df_filtered["diagnosa"].isin(top10_diag)
        ctab = pd.crosstab(
            df_filtered[mask]["diagnosa"],
            df_filtered[mask]["jenis_kelamin"]
        )
        st.dataframe(ctab)
    
    # Distribusi diagnosa per kelompok umur
    if "diagnosa" in df_filtered.columns and "kelompok_umur" in df_filtered.columns:
        st.markdown("### Distribusi Diagnosa (Top 5) per Kelompok Umur")
        top5_diag = df_filtered["diagnosa"].value_counts().head(5).index
        df_top5 = df_filtered[df_filtered["diagnosa"].isin(top5_diag)]
        pivot = pd.pivot_table(
            df_top5,
            index="diagnosa",
            columns="kelompok_umur",
            values="poli",
            aggfunc="count",
            fill_value=0
        )
        st.dataframe(pivot)

def page_pembiayaan(df_filtered, filter_info):
    st.subheader("💳 Analisis Pembiayaan")
    
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data untuk ditampilkan. Periksa kembali filter Anda.")
        return
    
    if "pembiayaan" not in df_filtered.columns:
        st.info("Kolom 'pembiayaan' tidak ditemukan di data.")
        return
    
    # Ringkasan umum
    st.markdown("### Ringkasan Kunjungan per Jenis Pembiayaan")
    ringkasan = (
        df_filtered
        .groupby("pembiayaan")
        .agg(
            jumlah_kunjungan=("pembiayaan", "size")
        )
        .sort_values("jumlah_kunjungan", ascending=False)
    )
    st.bar_chart(ringkasan["jumlah_kunjungan"])
    st.dataframe(ringkasan)
    
    # Pembiayaan per poli
    if "poli" in df_filtered.columns:
        st.markdown("### Matriks Poli x Pembiayaan")
        ctab = pd.crosstab(df_filtered["poli"], df_filtered["pembiayaan"])
        st.dataframe(ctab)

def page_data(df_filtered, filter_info):
    st.subheader("📄 Data Terfilter & Unduhan")
    
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data untuk ditampilkan. Periksa kembali filter Anda.")
        return
    
    st.markdown("### Data Terfilter")
    st.dataframe(df_filtered)
    
    # Download CSV
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download data terfilter (CSV)",
        data=csv,
        file_name="data_kunjungan_terfilter.csv",
        mime="text/csv"
    )
    
    # Ringkasan cepat (pivot sederhana)
    st.markdown("---")
    st.markdown("### Ringkasan Cepat (Pivot)")
    st.write("Pilih kombinasi kolom untuk membuat ringkasan cepat (jumlah kunjungan).")
    
    cols = df_filtered.columns.tolist()
    index_default = cols.index("poli") if "poli" in cols else 0
    index_col = st.selectbox("Index (baris)", options=cols, index=index_default)
    
    column_col = st.selectbox("Kolom (opsional)", options=["(Tidak ada)"] + cols)
    
    # Pilih kolom yang akan dihitung (count)
    value_default = "no_rm" if "no_rm" in cols else cols[0]
    value_index = cols.index(value_default)
    value_col = st.selectbox("Kolom untuk dihitung (count)", options=cols, index=value_index)
    
    if column_col == "(Tidak ada)":
        pivot = df_filtered.groupby(index_col)[value_col].count().reset_index(name="jumlah")
        st.dataframe(pivot)
    else:
        pivot = pd.pivot_table(
            df_filtered,
            index=index_col,
            columns=column_col,
            values=value_col,
            aggfunc="count",
            fill_value=0
        )
        st.dataframe(pivot)

def page_quality(df):
    st.subheader("🧹 Kualitas Data")
    
    if df is None or len(df) == 0:
        st.warning("Data belum tersedia. Upload file terlebih dahulu.")
        return
    
    st.markdown("### 1. Informasi Struktur Data")
    info = pd.DataFrame({
        "tipe_data": df.dtypes.astype(str),
        "jumlah_null": df.isna().sum(),
        "persen_null": (df.isna().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info)
    
    st.markdown("---")
    st.markdown("### 2. Cek Duplikasi")
    subset_cols = [c for c in ["no_rm", "tanggal_kunjungan", "poli"] if c in df.columns]
    if subset_cols:
        dupe_mask = df.duplicated(subset=subset_cols, keep=False)
        dupe_count = dupe_mask.sum()
        st.write(f"Jumlah baris duplikat (berdasarkan kolom {subset_cols}): **{dupe_count}**")
        if dupe_count > 0:
            with st.expander("Lihat data duplikat"):
                st.dataframe(df[dupe_mask].sort_values(subset_cols))
    else:
        st.info("Kolom kunci untuk cek duplikasi (no_rm, tanggal_kunjungan, poli) tidak lengkap.")
    
    st.markdown("---")
    st.markdown("### 3. Nilai Kategori Tidak Wajar (Outlier Kategori)")
    for col in ["poli", "jenis_kelamin", "pembiayaan"]:
        if col in df.columns:
            st.markdown(f"#### Kolom `{col}`")
            st.dataframe(df[col].value_counts(dropna=False))

# ========= MAIN =========

def main():
    # Terapkan filter & dapatkan data terfilter
    df_filtered, filter_info = apply_filters(None)
    
    # Sidebar: navigasi halaman
    st.sidebar.markdown("---")
    st.sidebar.header("📑 Navigasi Halaman")
    page = st.sidebar.radio(
        "Pilih halaman",
        options=[
            "Ringkasan Umum",
            "Analisis Kunjungan",
            "Analisis Penyakit",
            "Analisis Pembiayaan",
            "Data & Unduhan",
            "Kualitas Data"
        ]
    )
    
    # Tampilkan halaman sesuai pilihan
    if page == "Ringkasan Umum":
        page_overview(df_filtered, filter_info)
    elif page == "Analisis Kunjungan":
        page_kunjungan(df_filtered, filter_info)
    elif page == "Analisis Penyakit":
        page_penyakit(df_filtered, filter_info)
    elif page == "Analisis Pembiayaan":
        page_pembiayaan(df_filtered, filter_info)
    elif page == "Data & Unduhan":
        page_data(df_filtered, filter_info)
    elif page == "Kualitas Data":
        # Di sini dipakai data terfilter juga (boleh kamu ganti ke data mentah jika mau)
        page_quality(df_filtered)

if __name__ == "__main__":
    main()
