import streamlit as st
import pandas as pd
import numpy as np
import os
import re

from google import genai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="Dashboard Analisis Data Puskesmas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== TAMPILAN ==========
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Global background */
        body {
            background-color: #f5f7fb;
        }

        /* Kurangi padding default dan rapikan layout utama */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Style untuk judul utama */
        h1 {
            font-weight: 700 !important;
        }

        /* Metric cards terlihat seperti kartu */
        div[data-testid="metric-container"] {
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            background-color: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.6);
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }

        /* Sidebar background yang lebih lembut */
        section[data-testid="stSidebar"] {
            background-color: #f3f4f6;
        }

        /* Tabel/DataFrame dengan border lembut */
        div[data-testid="stDataFrame"] {
            border-radius: 0.5rem;
            border: 1px solid rgba(148, 163, 184, 0.4);
        }

        /* Garis pemisah halus */
        hr {
            margin: 0.75rem 0 1rem 0;
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

# ========= FUNGSI UTAMA PERSIAPAN DATA DAN Ai =========

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

# Nama kolom standar yang digunakan aplikasi
TARGET_COLS = [
    "tanggal_kunjungan",
    "no_rm",
    "umur",
    "jenis_kelamin",
    "poli",
    "diagnosa",
    "pembiayaan",
    "desa",
]

def _normalize_col_name(col: str) -> str:
    """
    Normalisasi nama kolom:
    - huruf kecil
    - hilangkan spasi, titik, garis, dll
    Contoh: 'Tgl Kunjungan' -> 'tglkunjungan'
    """
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]", "", col)
    return col


def _build_column_mapping(df_raw: pd.DataFrame) -> dict:
    """
    Mencoba memetakan nama kolom mentah ke nama standar
    berdasarkan beberapa alias yang umum dipakai di Puskesmas.
    """
    mapping = {}
    
    # Kamus alias -> nama standar
    alias_dict = {
        "tanggal_kunjungan": [
            "tanggalkunjungan", "tglkunjungan", "tanggal", "tgl", "visitdate",
            "tanggalkunjunganpasien"
        ],
        "no_rm": [
            "norm", "norekammedis", "norekam_medis", "no_rekam_medis",
            "rekammedis", "normbaru", "normlama"
        ],
        "umur": [
            "umur", "usia", "age", "umurth", "usiatahunan"
        ],
        "jenis_kelamin": [
            "jeniskelamin", "jk", "sex", "kelamin", "llp"
        ],
        "poli": [
            "poli", "politujuan", "unit", "unitlayanan", "poliklinik"
        ],
        "diagnosa": [
            "diagnosa", "diagnosautama", "dxutama", "diag", "icd10", "diagnosis"
        ],
        "pembiayaan": [
            "pembiayaan", "carabayar", "penjamin", "jaminan", "pembayar"
        ],
        "desa": [
            "desa", "alamatdesa", "kelurahan", "desakelurahan",
            "namadesa", "desapasien"
        ],
    }

    # Buat kamus cepat untuk lookup dari kol mentah -> norm key
    norm_cols = {col: _normalize_col_name(col) for col in df_raw.columns}

    for std_name, alias_list in alias_dict.items():
        for raw_col, norm_key in norm_cols.items():
            if norm_key in alias_list:
                mapping[raw_col] = std_name
                break

    return mapping


def _clean_value_jenis_kelamin(val):
    """
    Normalisasi nilai jenis kelamin ke 'Laki-Laki' / 'Perempuan'
    bila memungkinkan.
    """
    if pd.isna(val):
        return np.nan

    v = str(val).strip().lower()
    if v in ["l", "lk", "laki", "laki-laki", "laki laki", "male", "m", "1"]:
        return "Laki-Laki"
    if v in ["p", "pr", "perempuan", "wanita", "female", "f", "2"]:
        return "Perempuan"
    return str(val).title()


def _parse_umur(val):
    """
    Untuk kolom usia yang berbentuk teks, mis:
    '36 Thn 9 Bln 19 Hari' -> ambil angka tahun saja = 36.
    """
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return np.nan
    # fallback: coba langsung cast ke int/float
    try:
        return int(float(s))
    except Exception:
        return np.nan


def clean_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    - Mapping nama kolom mentah -> standar
    - Pilih hanya kolom yang relevan untuk aplikasi
    - Bersihkan tipe data dasar (tanggal, umur, jenis kelamin, teks)
    """
    if df_raw is None or df_raw.empty:
        return df_raw

    # 1) Mapping nama kolom
    col_map = _build_column_mapping(df_raw)
    df = df_raw.rename(columns=col_map).copy()

    # 2) Memilih kolom yang relevan (jika ada)
    keep_cols = [c for c in TARGET_COLS if c in df.columns]
    df = df[keep_cols].copy()

    # 3) Bersihkan tipe data dasar
    # Tanggal kunjungan
    if "tanggal_kunjungan" in df.columns:
        df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")

    # Umur
    if "umur" in df.columns:
        df["umur"] = df["umur"].apply(_parse_umur)

    # Jenis kelamin
    if "jenis_kelamin" in df.columns:
        df["jenis_kelamin"] = df["jenis_kelamin"].apply(_clean_value_jenis_kelamin)

    # String standar
    for col in ["poli", "diagnosa", "pembiayaan", "desa", "no_rm"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan})
            )

    return df

@st.cache_data
def load_data(file):
    """
    Membaca file CSV / Excel (termasuk Excel 97-2003 .xls) menjadi DataFrame mentah,
    dengan fallback otomatis jika file .xls sebenarnya adalah teks (CSV/TSV).
    """
    filename = file.name.lower()
    ext = filename.split(".")[-1]

    df_raw = None

    # 1. Kasus CSV biasa
    if ext == "csv":
        df_raw = pd.read_csv(file)

    # 2. Jika excel (xlsx/xls)
    elif ext in ["xlsx", "xls"]:
        try:
            if ext == "xlsx":
                df_raw = pd.read_excel(file, engine="openpyxl")
            else:
                # Mencoba sebagai Excel 97-2003
                df_raw = pd.read_excel(file, engine="xlrd")
        except Exception:
            # Jika gagal, cmencoba membaca (CSV/TSV)
            file.seek(0)
            try:
                # auto deteksi delimiter (bisa koma, titik koma, tab, dll)
                df_raw = pd.read_csv(file, sep=None, engine="python")
            except Exception:
                file.seek(0)
                # fallback terakhir: anggap tab-separated (tsv)
                df_raw = pd.read_csv(file, sep="\t")

    else:
        raise ValueError(f"Format file tidak didukung: .{ext}")

    # Cleaning & memilih kolom relevan
    df_clean = clean_raw_data(df_raw)
    return df_clean, df_raw

def preprocess_data(df):
    """Membersihkan dan menambah kolom turunan."""
    if df is None or df.empty:
        return df

    # Normalisasi nama kolom (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # DROP semua kolom alamat yang bukan desa
    # contoh yang didrop: 'alamat', 'alamat_lengkap', 'alamat_rumah'
    # contoh yang TIDAK didrop: 'alamat_desa', 'desa', 'desa_pasien'
    cols_to_drop = [c for c in df.columns if ("alamat" in c and "desa" not in c)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Pastikan ada kolom 'desa' jika ada kolom lain yang mengandung kata 'desa'
    if "desa" not in df.columns:
        for c in df.columns:
            if "desa" in c.lower():
                df["desa"] = df[c]
                break
    
    # Konversi tanggal kunjungan
    if "tanggal_kunjungan" in df.columns:
        df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
    
    # Memastikan umur numerik jika ada
    if "umur" in df.columns:
        df["umur"] = pd.to_numeric(df["umur"], errors="coerce")
    
    # Pengelompokan umur
    if "umur" in df.columns:
        bins = [0, 1, 5, 15, 25, 45, 60, 200]
        labels = ["<1 th", "1-4 th", "5-14 th", "15-24 th", "25-44 th", "45-59 th", "60+ th"]
        df["kelompok_umur"] = pd.cut(df["umur"], bins=bins, labels=labels, right=False)
    
    # Menambahkan kolom tahun & bulan (angka & nama)
    if "tanggal_kunjungan" in df.columns:
        df["tahun"] = df["tanggal_kunjungan"].dt.year
        df["bulan"] = df["tanggal_kunjungan"].dt.month
        df["nama_bulan"] = df["tanggal_kunjungan"].dt.strftime("%b")
    
    # Standarisasi beberapa kolom teks jika ada
    for col in ["poli", "jenis_kelamin", "pembiayaan", "diagnosa", "desa"]:
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
        st.markdown("## 🏥 Data Kunjungan Puskesmas")
        uploaded_file = st.file_uploader(
            "Upload data kunjungan (CSV / Excel)",
            type=["csv", "xlsx", "xls"],
        )
        
        if uploaded_file is None:
            st.info("Silakan upload file **CSV/Excel** yang berisi data kunjungan untuk mulai analisis.")
            return None, None

        # >>> BACA DAN CLEANING DATA <<<
        df_clean, df_raw = load_data(uploaded_file)
        df = preprocess_data(df_clean)
        
        st.success("Data berhasil dimuat ✅")
        st.caption(f"📊 {len(df):,} baris • {len(df.columns)} kolom".replace(",", "."))

        # (opsional) preview data mentah dan hasil cleaning
        with st.expander("Lihat contoh data mentah & hasil cleaning"):
            st.caption("Contoh 5 baris data mentah (raw):")
            st.dataframe(df_raw.head(5))
            st.caption("Contoh 5 baris data setelah cleaning & standarisasi:")
            st.dataframe(df.head(5))

        # Inisialisasi variabel filter
        date_range = None
        tahun_pilihan = None
        poli_pilihan = None
        jk_pilihan = None
        bayar_pilihan = None
        kelompok_umur_pilihan = None
        desa_pilihan = None
        
        st.markdown("### 🔍 Filter Data")

        # ---------- FILTER WKTU DAN POLI ----------
        with st.expander("Filter Waktu & Poli", expanded=True):
            # Filter tanggal
            if "tanggal_kunjungan" in df.columns:
                min_date = df["tanggal_kunjungan"].min()
                max_date = df["tanggal_kunjungan"].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = st.date_input(
                        "Rentang tanggal kunjungan",
                        value=[min_date.date(), max_date.date()],
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                    )

            # Filter tahun
            if "tahun" in df.columns:
                tahun_unik = sorted(df["tahun"].dropna().unique())
                if len(tahun_unik) > 0:
                    tahun_pilihan = st.multiselect(
                        "Filter tahun",
                        options=tahun_unik,
                        default=tahun_unik,
                    )

            # Filter poli
            if "poli" in df.columns:
                poli_unik = sorted(df["poli"].dropna().unique())
                poli_pilihan = st.multiselect(
                    "Filter poli",
                    options=poli_unik,
                    default=poli_unik,
                )

        # ---------- FILTER DEMOGRAFI (TERMASUK DESA) ----------
        with st.expander("Filter Demografi", expanded=False):
            # Filter jenis kelamin
            if "jenis_kelamin" in df.columns:
                jk_unik = sorted(df["jenis_kelamin"].dropna().unique())
                jk_pilihan = st.multiselect(
                    "Filter jenis kelamin",
                    options=jk_unik,
                    default=jk_unik,
                )

            # Filter kelompok umur
            if "kelompok_umur" in df.columns:
                ku_unik = [x for x in df["kelompok_umur"].dropna().unique()]
                kelompok_umur_pilihan = st.multiselect(
                    "Filter kelompok umur",
                    options=ku_unik,
                    default=ku_unik,
                )

            # Filter desa/kelurahan
            if "desa" in df.columns:
                desa_unik = sorted(df["desa"].dropna().unique())
                desa_pilihan = st.multiselect(
                    "Filter desa/kelurahan",
                    options=desa_unik,
                    default=desa_unik,
                )

        # ---------- FILTER PEMBAYARAN ----------
        with st.expander("Filter Pembiayaan", expanded=False):
            if "pembiayaan" in df.columns:
                bayar_unik = sorted(df["pembiayaan"].dropna().unique())
                bayar_pilihan = st.multiselect(
                    "Filter pembiayaan",
                    options=bayar_unik,
                    default=bayar_unik,
                )

        st.markdown("---")
        st.caption("💡 Tip: Sesuaikan filter untuk melihat pola kunjungan berdasarkan segmen tertentu.")

    # ================== MENGAPLIKASIKAN FILTER DI LUAR SIDEBAR ==================
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
        df_filtered = df_filtered[df_filtered["desa"].isin(desa_pilihan)]
    
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


def show_active_filters(filter_info):
    """Tampilkan ringkasan filter aktif di bagian atas halaman."""
    if not filter_info:
        return

    chips = []

    if filter_info.get("poli"):
        chips.append("Poli: " + ", ".join(map(str, filter_info["poli"])))
    if filter_info.get("jenis_kelamin"):
        chips.append("JK: " + ", ".join(map(str, filter_info["jenis_kelamin"])))
    if filter_info.get("kelompok_umur"):
        chips.append(
            "Kelompok umur: "
            + ", ".join(map(lambda x: str(x), filter_info["kelompok_umur"]))
        )
    if filter_info.get("pembiayaan"):
        chips.append("Pembiayaan: " + ", ".join(map(str, filter_info["pembiayaan"])))
    if filter_info.get("desa"):
        chips.append("Desa: " + ", ".join(map(str, filter_info["desa"])))

    if chips:
        st.caption("🎯 **Filter aktif:** " + " | ".join(chips))


# ========= HALAMAN NIH =========

def page_overview(df_filtered, filter_info):
    st.subheader("📌 Ringkasan Umum")
    show_active_filters(filter_info)
    
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
        st.markdown("### 📈 Tren Kunjungan per Bulan")
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
        st.markdown("### 🏥 Distribusi Kunjungan per Poli")
        poli_counts = df_filtered["poli"].value_counts().sort_values(ascending=False)
        if len(poli_counts) > 0:
            st.bar_chart(poli_counts)
            st.dataframe(poli_counts.rename("Jumlah Kunjungan"))
    
    # Diagnosa ter baek
    if "diagnosa" in df_filtered.columns:
        st.markdown("### 🦠 Top 10 Diagnosa Terbanyak")
        top_diag = df_filtered["diagnosa"].value_counts().head(10)
        st.bar_chart(top_diag)
        st.dataframe(top_diag.rename("Jumlah Kunjungan"))


def page_kunjungan(df_filtered, filter_info):
    st.subheader("👥 Analisis Kunjungan & Pasien")
    show_active_filters(filter_info)
    
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
    show_active_filters(filter_info)
    
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data untuk ditampilkan. Periksa kembali filter Anda.")
        return
    
    # Pilih poli (opsional) untuk fokus menganalisa
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
    show_active_filters(filter_info)
    
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


def page_ml(df_filtered, filter_info):
    st.subheader("🧠 Prediksi Risiko Kejadian di Masa Depan")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data setelah filter diterapkan. Silakan atur filter terlebih dahulu.")
        return

    if "tanggal_kunjungan" not in df_filtered.columns:
        st.warning("Kolom 'tanggal_kunjungan' tidak ditemukan. Halaman prediksi risiko membutuhkan data tanggal.")
        return

    # Memastikan tanggal dalam format datetime
    df = df_filtered.copy()
    df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
    df = df.dropna(subset=["tanggal_kunjungan"])

    if len(df) == 0:
        st.warning("Semua nilai 'tanggal_kunjungan' tidak valid.")
        return

    st.markdown(
        """
        Halaman ini menggunakan **model Machine Learning sederhana** untuk:
        
        - Menganalisis **tren kunjungan historis** pada suatu diagnosa atau poli.  
        - Memprediksi **risiko lonjakan kasus** dalam beberapa bulan ke depan, jika **tidak ada tindakan pencegahan**.  
        - Menampilkan **grafik tren** dan **probabilitas risiko** per bulan.  
        """
    )

    # Pilih fokus analisa: Diagnosa atau Poli
    fokus = st.radio(
        "Pilih fokus analisis",
        options=["Diagnosa", "Poli"],
        horizontal=True,
    )

    kolom_fokus = "diagnosa" if fokus == "Diagnosa" else "poli"
    if kolom_fokus not in df.columns:
        st.warning(f"Kolom '{kolom_fokus}' tidak ditemukan di data.")
        return

    # Pilih item spesifik (diagnosa/poli)
    top_items = (
        df[kolom_fokus]
        .value_counts()
        .head(30)
    )
    if len(top_items) == 0:
        st.warning(f"Tidak ada nilai pada kolom '{kolom_fokus}'.")
        return

    pilihan_item = st.selectbox(
        f"Pilih {fokus.lower()} yang akan dianalisis risikonya",
        options=top_items.index.tolist(),
        format_func=lambda x: f"{x} ({top_items[x]} kunjungan)",
    )

    df_item = df[df[kolom_fokus] == pilihan_item].copy()

    if len(df_item) < 10:
        st.warning(f"Data untuk {fokus.lower()} '{pilihan_item}' terlalu sedikit untuk analisis risiko.")
        return

    # Agregasi per bulan
    df_item["periode"] = df_item["tanggal_kunjungan"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df_item.groupby("periode")
        .size()
        .reset_index(name="jumlah_kunjungan")
        .sort_values("periode")
    )

    if len(monthly) < 6:
        st.warning("Riwayat data bulanan kurang dari 6 titik. Prediksi akan kurang stabil.")
    
    st.markdown("### 📈 Tren Historis Kunjungan per Bulan")
    st.line_chart(
        monthly.set_index("periode")["jumlah_kunjungan"],
        height=260,
    )

    # Siapkan fitur untuk model
    monthly["tahun"] = monthly["periode"].dt.year
    monthly["bulan"] = monthly["periode"].dt.month
    monthly["t"] = range(len(monthly))  # indeks waktu

    # Definisikan 'lonjakan' sebagai bulan dengan kunjungan >= persentil tertentu
    persentil = st.slider(
        "Ambang definisi 'lonjakan kasus' (persentil historis)",
        min_value=50,
        max_value=95,
        value=75,
        step=5,
        help="Bulan dengan jumlah kunjungan di atas persentil ini akan dianggap sebagai 'lonjakan kasus'.",
    )
    threshold = np.percentile(monthly["jumlah_kunjungan"], persentil)
    monthly["is_lonjakan"] = (monthly["jumlah_kunjungan"] >= threshold).astype(int)

    st.caption(
        f"Ambang lonjakan untuk {fokus.lower()} **{pilihan_item}**: "
        f"≥ {threshold:.0f} kunjungan/bulan (persentil {persentil})."
    )

    # Fitur & label
    X = monthly[["t", "tahun", "bulan"]]
    y = monthly["is_lonjakan"]

    # Jika semua 0 atau semua 1 -> tidak bisa klasifikasi
    if y.nunique() < 2:
        st.warning(
            "Riwayat data tidak memiliki variasi cukup untuk membedakan bulan 'lonjakan' dan 'tidak lonjakan'. "
            "Coba pilih diagnosa/poli lain atau ubah persentil ambang lonjakan."
        )
        return

    # Model klasifikasi risiko lonjakan: RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    # Split train/tes jika data cukup
    if len(monthly) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.markdown("### 📊 Kinerja Model (Data Historis)")
        colm1, colm2 = st.columns(2)
        colm1.metric("Akurasi pada data uji", f"{acc*100:.1f}%")
        colm2.metric("Jumlah sampel uji", len(y_test))

        with st.expander("Detail distribusi kelas & laporan klasifikasi"):
            st.write("Distribusi kelas (0 = tidak lonjakan, 1 = lonjakan):")
            st.write(y.value_counts())
            st.text(classification_report(y_test, y_pred))
    else:
        # Data sangat sedikit -> latih di seluruh data tanpa evaluasi
        model.fit(X, y)
        st.info(
            "Data historis kurang dari 10 titik. Model dilatih pada seluruh data tanpa pemisahan train/test, "
            "sehingga metrik kinerja tidak ditampilkan."
        )

    # Latih ulang model pada seluruh data sebelum dipakai memprediksi kedepannya
    model.fit(X, y)

    # Horizon prediksi
    horizon = st.slider(
        "Prediksi berapa bulan ke depan?",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
    )

    # Buat periode masa depan
    last_period = monthly["periode"].max()
    future_periods = pd.date_range(
        start=last_period + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    future_df = pd.DataFrame({"periode": future_periods})
    future_df["tahun"] = future_df["periode"].dt.year
    future_df["bulan"] = future_df["periode"].dt.month
    future_df["t"] = range(monthly["t"].max() + 1, monthly["t"].max() + 1 + len(future_df))

    # Prediksi probabilitas lonjakan
    proba_future = model.predict_proba(future_df[["t", "tahun", "bulan"]])[:, 1]
    future_df["prob_lonjakan"] = proba_future

    st.markdown("### 🔮 Probabilitas Risiko Lonjakan di Masa Depan")
    st.line_chart(
        future_df.set_index("periode")[["prob_lonjakan"]],
        height=260,
    )

    st.markdown("### 📅 Rincian Probabilitas per Bulan")
    detail_prob = future_df[["periode", "prob_lonjakan"]].copy()
    detail_prob["prob_lonjakan_%"] = (detail_prob["prob_lonjakan"] * 100).round(1)
    detail_prob = detail_prob.rename(
        columns={
            "periode": "Periode (bulan)",
            "prob_lonjakan": "Probabilitas lonjakan (0-1)",
            "prob_lonjakan_%": "Probabilitas lonjakan (%)",
        }
    )
    st.dataframe(detail_prob)

    # Statistik ringkas
    max_idx = detail_prob["Probabilitas lonjakan (0-1)"].idxmax()
    max_row = detail_prob.loc[max_idx]
    rata2 = detail_prob["Probabilitas lonjakan (0-1)"].mean()
    tinggi = detail_prob[detail_prob["Probabilitas lonjakan (0-1)"] >= 0.7]

    st.markdown("### 📌 Ringkasan Risiko")
    colr1, colr2, colr3 = st.columns(3)
    colr1.metric(
        "Probabilitas rata-rata",
        f"{rata2*100:.1f}%",
    )
    colr2.metric(
        "Probabilitas tertinggi",
        f"{max_row['Probabilitas lonjakan (%)']:.1f}%",
        help=f"Periode: {max_row['Periode (bulan)'].strftime('%b %Y')}",
    )

    if len(tinggi) > 0:
        with st.expander("Lihat bulan berisiko tinggi (≥70%)"):
            st.dataframe(tinggi)


def page_data(df_filtered, filter_info):
    st.subheader("📄 Data Terfilter & Unduhan")
    show_active_filters(filter_info)
    
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


def build_ai_context(df_filtered, filter_info):
    """Menyusun ringkasan teks dari data terfilter untuk dikirim ke AI."""
    if df_filtered is None or len(df_filtered) == 0:
        return "Tidak ada data setelah filter diterapkan."

    parts = []

    # Ringkasan dasar
    total_kunjungan = len(df_filtered)
    total_pasien = df_filtered["no_rm"].nunique() if "no_rm" in df_filtered.columns else None

    parts.append(f"Total kunjungan: {total_kunjungan}")
    if total_pasien is not None:
        parts.append(f"Perkiraan jumlah pasien unik: {total_pasien}")

    # Info rentang tanggal
    if "tanggal_kunjungan" in df_filtered.columns:
        min_tgl = df_filtered["tanggal_kunjungan"].min()
        max_tgl = df_filtered["tanggal_kunjungan"].max()
        if pd.notna(min_tgl) and pd.notna(max_tgl):
            parts.append(f"Rentang tanggal kunjungan: {min_tgl.date()} s.d. {max_tgl.date()}")

    # Filter yang sedang aktif (ringkas)
    if filter_info is not None:
        aktif = []
        if filter_info.get("poli"):
            aktif.append(f"Poli: {', '.join(map(str, filter_info['poli']))}")
        if filter_info.get("jenis_kelamin"):
            aktif.append(f"Jenis kelamin: {', '.join(map(str, filter_info['jenis_kelamin']))}")
        if filter_info.get("pembiayaan"):
            aktif.append(f"Pembiayaan: {', '.join(map(str, filter_info['pembiayaan']))}")
        if filter_info.get("kelompok_umur"):
            aktif.append(
                "Kelompok umur: "
                + ", ".join(map(lambda x: str(x), filter_info["kelompok_umur"]))
            )
        if filter_info.get("desa"):
            aktif.append(f"Desa/kelurahan: {', '.join(map(str, filter_info['desa']))}")

        if aktif:
            parts.append("Filter aktif: " + " | ".join(aktif))

    # Top poli
    if "poli" in df_filtered.columns:
        top_poli = df_filtered["poli"].value_counts().head(5)
        teks_top_poli = "; ".join(
            [f"{idx}: {val} kunjungan" for idx, val in top_poli.items()]
        )
        parts.append("Top 5 poli berdasarkan jumlah kunjungan: " + teks_top_poli)

    # Top diagnosa
    if "diagnosa" in df_filtered.columns:
        top_diag = df_filtered["diagnosa"].value_counts().head(5)
        teks_top_diag = "; ".join(
            [f"{idx}: {val} kunjungan" for idx, val in top_diag.items()]
        )
        parts.append("Top 5 diagnosa terbanyak: " + teks_top_diag)

    # Distribusi pembiayaan (opsional)
    if "pembiayaan" in df_filtered.columns:
        top_biaya = df_filtered["pembiayaan"].value_counts()
        teks_biaya = "; ".join(
            [f"{idx}: {val} kunjungan" for idx, val in top_biaya.items()]
        )
        parts.append("Distribusi pembiayaan: " + teks_biaya)

    context_text = "\n".join(parts)
    return context_text


def page_ai_assistant(df_filtered, filter_info, client):
    st.subheader("🤖 Asisten AI untuk Analisis Puskesmas")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data setelah filter diterapkan. Silakan atur filter terlebih dahulu.")
        return

    if client is None:
        st.error("Client Gemini belum siap karena API key tidak ditemukan.")
        return

    st.markdown(
        """
        Asisten AI ini akan membantu memberi **interpretasi** dan **saran program/kebijakan**
        berdasarkan data yang sudah difilter di sidebar.

        1. Atur filter data di sidebar (poli, tanggal, pembiayaan, dsb).
        2. Tulis pertanyaan atau masalah yang ingin dibahas.
        3. Tekan tombol **Dapatkan Saran AI**.
        """
    )

    # Hitung konteks satu kali
    context_text = build_ai_context(df_filtered, filter_info)

    # Tampilkan ringkasan konteks yang akan dikirim
    with st.expander("Lihat ringkasan data yang dikirim ke AI"):
        st.text(context_text)

    # Input pertanyaan user
    user_question = st.text_area(
        "Tulis pertanyaan / masalah yang ingin dibantu AI",
        placeholder=(
            "Contoh:\n"
            "- Apa kemungkinan masalah utama kesehatan di wilayah ini?\n"
            "- Program apa yang sebaiknya diprioritaskan berdasarkan pola penyakit?\n"
            "- Bagaimana rekomendasi intervensi untuk menurunkan kasus ISPA pada balita?"
        ),
        height=150,
    )

    if st.button("Dapatkan Saran AI"):
        if not user_question.strip():
            st.warning("Silakan isi pertanyaan terlebih dahulu.")
            return

        # Contoh prompt untuk Ai
        prompt = f"""
Anda adalah analis kesehatan masyarakat yang membantu Puskesmas.

Berikut ringkasan data kunjungan yang sudah difilter:

{context_text}

Tugas Anda:
1. Jelaskan interpretasi pola kunjungan / penyakit / pembiayaan dari data tersebut.
2. Jelaskan potensi masalah kesehatan utama yang perlu diwaspadai.
3. Berikan rekomendasi program, kegiatan, atau kebijakan yang bisa dilakukan Puskesmas.
4. Sesuaikan jawaban dengan konteks fasilitas pelayanan dasar di Indonesia.

Pertanyaan spesifik dari pengguna:
\"\"\"{user_question}\"\"\"

Jawab dengan bahasa Indonesia yang jelas, terstruktur (gunakan poin-poin), dan praktis untuk pengambilan keputusan.
"""

        with st.spinner("Menghubungi AI dan menyusun rekomendasi..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                ai_text = response.text
                st.markdown("### 💡 Rekomendasi dari Asisten AI")
                st.markdown(ai_text)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memanggil API Gemini: {e}")


# ========= MAIN =========

def main():
    # Menerapkan filter & dapatkan data terfilter
    df_filtered, filter_info = apply_filters(None)

    # Buat client Gemini (sekali saja, cache_resource)
    client = get_gemini_client()

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
            "Kualitas Data",
            "Model ML",
            "Asisten AI",
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
        page_quality(df_filtered)
    elif page == "Model ML":
        page_ml(df_filtered, filter_info)
    elif page == "Asisten AI":
        page_ai_assistant(df_filtered, filter_info, client)


if __name__ == "__main__":
    main()
