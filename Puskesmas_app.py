import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import io
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
from datetime import datetime, date
import tempfile

# ─── Machine Learning & Forecasting Libraries ───────────────────────────────
import google.generativeai as genai
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# XGBoost & scikit-learn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Dashboard Puskesmas – Ensemble AI Forecasting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    .block-container { padding: 1.5rem 2rem 3rem 2rem; }
    h1 { font-weight: 700 !important; }
    div[data-testid="metric-container"] {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        background-color: var(--secondary-background-color);
        border: 1px solid var(--faded-text-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    }
    .model-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.4rem;
    }
    .badge-prophet  { background:#dbeafe; color:#1d4ed8; }
    .badge-xgb      { background:#dcfce7; color:#166534; }
    .badge-sarima   { background:#fef3c7; color:#92400e; }
    .badge-ensemble { background:#f3e8ff; color:#6b21a8; }
    .badge-winner   { background:#fecaca; color:#991b1b; }
    .accuracy-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: rgba(59,130,246,0.07);
        border: 1px solid rgba(59,130,246,0.25);
        margin-bottom: 1rem;
    }
    .highlight-estimasi {
        color: #1d4ed8 !important;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ============================================================
# HEADER
# ============================================================
st.title("📊 Dashboard Analisis Data Puskesmas")
st.markdown("""
Dashboard ini membantu menganalisis **data kunjungan Puskesmas** dengan:
- Filter interaktif: Poli, Diagnosa, Umur, Desa, Pembiayaan
- **Ensemble Forecasting:** Prophet + XGBoost + SARIMA + Auto-Selection terbaik
- **Disease Dominance Detection** per musim/bulan
- **Evaluasi Akurasi** model (MAE, RMSE, MAPE) ditampilkan ke pengguna

Untuk data cleaning dapat menggunakan aplikasi berikut:
https://data-cleaning-app-for-puskesmasapp.streamlit.app/

⬅️ **Mulai dengan meng-upload file data di sidebar.**
""")

# ============================================================
# GEMINI CLIENT
# ============================================================
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
        st.warning("⚠️ API key Gemini belum diset. Fitur AI Agent tidak aktif.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Gagal inisialisasi Gemini: {e}")
        return False

# ============================================================
# HELPER UMUM
# ============================================================
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()


TARGET_COLS = ["tanggal_kunjungan","no_rm","umur","jenis_kelamin","poli","diagnosa","pembiayaan","desa"]

def _normalize_col_name(col):
    col = str(col).strip().lower()
    return re.sub(r"[^a-z0-9]", "", col)

def _build_column_mapping(df_raw):
    mapping = {}
    alias_dict = {
        "tanggal_kunjungan": ["tanggalkunjungan","tglkunjungan","tanggal","tgl","visitdate"],
        "no_rm":    ["norm","norekammedis","norekam_medis","no_rekam_medis","rekammedis"],
        "umur":     ["umur","usia","age","umurth"],
        "jenis_kelamin": ["jeniskelamin","jk","sex","kelamin","llp"],
        "poli":     ["poli","politujuan","unit","unitlayanan","poliklinik"],
        "diagnosa": ["diagnosa","diagnosautama","dxutama","diag","icd10","diagnosis"],
        "pembiayaan":["pembiayaan","carabayar","penjamin","jaminan","pembayar"],
        "desa":     ["desa","alamatdesa","kelurahan","desakelurahan","namadesa"],
    }
    norm_cols = {col: _normalize_col_name(col) for col in df_raw.columns}
    for std_name, aliases in alias_dict.items():
        for raw_col, norm_key in norm_cols.items():
            if norm_key in aliases:
                mapping[raw_col] = std_name
                break
    return mapping

def _clean_jk(val):
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
    keep = [c for c in TARGET_COLS if c in df.columns]
    df = df[keep].copy()
    if "tanggal_kunjungan" in df.columns:
        df["tanggal_kunjungan"] = pd.to_datetime(df["tanggal_kunjungan"], errors="coerce")
    if "umur" in df.columns:
        df["umur"] = df["umur"].apply(_parse_umur)
    if "jenis_kelamin" in df.columns:
        df["jenis_kelamin"] = df["jenis_kelamin"].apply(_clean_jk)
    for col in ["poli","diagnosa","pembiayaan","desa","no_rm"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
    return df

@st.cache_data
def load_data(file):
    ext = file.name.lower().split(".")[-1]
    if ext == "csv":
        df_raw = pd.read_csv(file)
    else:
        try:    df_raw = pd.read_excel(file, engine="openpyxl")
        except: df_raw = pd.read_excel(file, engine="xlrd")
    return clean_raw_data(df_raw), df_raw

def preprocess_data(df):
    if df is None or df.empty: return df
    df.columns = [c.strip().lower() for c in df.columns]
    if "umur" in df.columns:
        df["umur"] = pd.to_numeric(df["umur"], errors="coerce")
        bins   = [0,1,5,15,25,45,60,200]
        labels = ["<1 th","1-4 th","5-14 th","15-24 th","25-44 th","45-59 th","60+ th"]
        df["kelompok_umur"] = pd.cut(df["umur"], bins=bins, labels=labels, right=False)
    if "tanggal_kunjungan" in df.columns:
        df["tahun"]      = df["tanggal_kunjungan"].dt.year
        df["bulan"]      = df["tanggal_kunjungan"].dt.month
        df["nama_bulan"] = df["tanggal_kunjungan"].dt.strftime("%b")
        df["minggu"]     = df["tanggal_kunjungan"].dt.isocalendar().week.astype(int)
        df["hari_ke"]    = df["tanggal_kunjungan"].dt.dayofyear
        # Musim hujan Indonesia: Nov-Apr → 1, kemarau: Mei-Okt → 0
        df["musim_hujan"] = df["bulan"].apply(lambda m: 1 if m in [11,12,1,2,3,4] else 0)
    for col in ["poli","jenis_kelamin","pembiayaan","diagnosa","desa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

# ============================================================
# SIDEBAR + FILTER
# ============================================================
def apply_filters(_):
    with st.sidebar:
        st.markdown("## 🏥 Data Kunjungan")
        uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])
        if uploaded_file is None:
            st.info("Silakan upload file CSV/Excel.")
            return None, None
        df_clean, _ = load_data(uploaded_file)
        df = preprocess_data(df_clean)
        st.success("Data berhasil dimuat ✅")
        st.caption(f"📊 {len(df):,} baris · {len(df.columns)} kolom".replace(",","."))

        st.markdown("### 🔍 Filter Data")
        date_range = poli_pilihan = jk_pilihan = bayar_pilihan = None
        kelompok_umur_pilihan = desa_pilihan = kecuali_penyakit = None

        with st.expander("Waktu & Poli", expanded=True):
            if "tanggal_kunjungan" in df.columns:
                mn, mx = df["tanggal_kunjungan"].min(), df["tanggal_kunjungan"].max()
                if pd.notna(mn) and pd.notna(mx):
                    date_range = st.date_input("Rentang Tanggal", value=[mn.date(), mx.date()])
            if "poli" in df.columns:
                poli_pilihan = st.multiselect("Poli", sorted(df["poli"].dropna().unique()))

        with st.expander("Filter Lainnya", expanded=False):
            if "jenis_kelamin" in df.columns:
                jk_pilihan = st.multiselect("Jenis Kelamin", sorted(df["jenis_kelamin"].dropna().unique()))
            if "kelompok_umur" in df.columns:
                kelompok_umur_pilihan = st.multiselect("Kelompok Umur", df["kelompok_umur"].dropna().unique())
            if "desa" in df.columns:
                desa_pilihan = st.multiselect("Desa", sorted(df["desa"].dropna().unique()))
            if "pembiayaan" in df.columns:
                bayar_pilihan = st.multiselect("Pembiayaan", sorted(df["pembiayaan"].dropna().unique()))
            if "diagnosa" in df.columns:
                kecuali_penyakit = st.multiselect(
                    "❌ Kecualikan Penyakit",
                    sorted(df["diagnosa"].dropna().unique()),
                    help="Penyakit ini tidak diikutkan dalam analisis."
                )

        df_f = df.copy()
        if date_range and len(date_range)==2:
            df_f = df_f[(df_f["tanggal_kunjungan"].dt.date >= date_range[0]) &
                        (df_f["tanggal_kunjungan"].dt.date <= date_range[1])]
        if poli_pilihan:           df_f = df_f[df_f["poli"].isin(poli_pilihan)]
        if jk_pilihan:             df_f = df_f[df_f["jenis_kelamin"].isin(jk_pilihan)]
        if bayar_pilihan:          df_f = df_f[df_f["pembiayaan"].isin(bayar_pilihan)]
        if kelompok_umur_pilihan:  df_f = df_f[df_f["kelompok_umur"].isin(kelompok_umur_pilihan)]
        if desa_pilihan:           df_f = df_f[df_f["desa"].isin(desa_pilihan)]
        if kecuali_penyakit:       df_f = df_f[~df_f["diagnosa"].isin(kecuali_penyakit)]

        return df_f, {
            "poli": poli_pilihan, "jenis_kelamin": jk_pilihan,
            "pembiayaan": bayar_pilihan, "kelompok_umur": kelompok_umur_pilihan,
            "desa": desa_pilihan, "penyakit_dikecualikan": kecuali_penyakit,
        }

def show_active_filters(fi):
    if not fi: return
    chips = [f"{k.replace('_',' ').title()}: {', '.join(map(str,v))}" for k,v in fi.items() if v]
    if chips: st.caption("🎯 **Filter aktif:** " + " | ".join(chips))

# ============================================================
# ═══════════  ENSEMBLE FORECASTING ENGINE  ═══════════════════
# ============================================================

# ── Feature Engineering ─────────────────────────────────────
def build_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur temporal & lag ke data mingguan."""
    df = weekly.copy().sort_values("ds").reset_index(drop=True)
    df["week_of_year"]  = df["ds"].dt.isocalendar().week.astype(int)
    df["month"]         = df["ds"].dt.month
    df["quarter"]       = df["ds"].dt.quarter
    df["year"]          = df["ds"].dt.year
    df["musim_hujan"]   = df["month"].apply(lambda m: 1 if m in [11,12,1,2,3,4] else 0)
    df["is_ramadan"]    = 0  # placeholder – bisa diisi manual
    # Lag features
    for lag in [1, 2, 3, 4]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    # Rolling mean
    df["roll_mean_4"]  = df["y"].shift(1).rolling(4).mean()
    df["roll_mean_8"]  = df["y"].shift(1).rolling(8).mean()
    df["roll_std_4"]   = df["y"].shift(1).rolling(4).std()
    return df.dropna().reset_index(drop=True)


# ── Metrik Evaluasi ──────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


# ── Prophet ──────────────────────────────────────────────────
def run_prophet(train_df, periods, freq="W-MON"):
    """Fit Prophet dan kembalikan forecast + metrics."""
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.05,
    )
    # Tambah regressor musim hujan
    train_ext = train_df.copy()
    train_ext["musim_hujan"] = train_ext["ds"].dt.month.apply(lambda mo: 1 if mo in [11,12,1,2,3,4] else 0)
    m.add_regressor("musim_hujan")
    m.fit(train_ext)

    future = m.make_future_dataframe(periods=periods, freq=freq)
    future["musim_hujan"] = future["ds"].dt.month.apply(lambda mo: 1 if mo in [11,12,1,2,3,4] else 0)
    fc = m.predict(future)

    # Hitung metrik pada data training
    merged = train_ext.merge(fc[["ds","yhat"]], on="ds")
    metrics = eval_metrics(merged["y"], merged["yhat"])
    return fc, metrics, m


# ── XGBoost ──────────────────────────────────────────────────
def run_xgboost(train_df, periods, freq="W-MON"):
    """Fit XGBoost dengan feature engineering dan prediksi iteratif."""
    feat_df = build_features(train_df)
    if len(feat_df) < 10:
        return None, None

    feat_cols = ["week_of_year","month","quarter","year","musim_hujan",
                 "lag_1","lag_2","lag_3","lag_4",
                 "roll_mean_4","roll_mean_8","roll_std_4"]
    X_train = feat_df[feat_cols]
    y_train = feat_df["y"]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # In-sample metrics
    y_pred_train = model.predict(X_train)
    metrics = eval_metrics(y_train, y_pred_train)

    # Iterative future prediction
    last_date   = train_df["ds"].max()
    history_y   = list(train_df["y"].values)
    future_rows = []
    for i in range(periods):
        next_date = last_date + pd.Timedelta(weeks=i+1)
        row = {
            "week_of_year": int(next_date.isocalendar()[1]),
            "month":        next_date.month,
            "quarter":      (next_date.month - 1) // 3 + 1,
            "year":         next_date.year,
            "musim_hujan":  1 if next_date.month in [11,12,1,2,3,4] else 0,
            "lag_1": history_y[-1],
            "lag_2": history_y[-2] if len(history_y) >= 2 else history_y[-1],
            "lag_3": history_y[-3] if len(history_y) >= 3 else history_y[-1],
            "lag_4": history_y[-4] if len(history_y) >= 4 else history_y[-1],
            "roll_mean_4": np.mean(history_y[-4:]),
            "roll_mean_8": np.mean(history_y[-8:]),
            "roll_std_4":  np.std(history_y[-4:]),
        }
        X_row  = pd.DataFrame([row])[feat_cols]
        y_next = float(model.predict(X_row)[0])
        history_y.append(max(0, y_next))
        future_rows.append({"ds": next_date, "yhat": max(0, y_next)})

    # Gabung dengan in-sample predictions
    train_preds = pd.DataFrame({"ds": feat_df["ds"], "yhat": y_pred_train.clip(min=0)})
    future_df   = pd.DataFrame(future_rows)
    all_preds   = pd.concat([train_preds, future_df], ignore_index=True)
    return all_preds, metrics


# ── SARIMA ───────────────────────────────────────────────────
def run_sarima(train_df, periods):
    """Fit SARIMA(1,1,1)(1,1,0,52) dan prediksi."""
    ts = train_df.set_index("ds")["y"].asfreq("W-MON").ffill()
    try:
        model = SARIMAX(
            ts, order=(1,1,1), seasonal_order=(1,1,0,52),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fit   = model.fit(disp=False, maxiter=200)

        # In-sample
        y_pred_train = fit.fittedvalues.clip(lower=0)
        metrics = eval_metrics(ts.values, y_pred_train.values)

        # Future
        fc_obj    = fit.get_forecast(steps=periods)
        fc_mean   = fc_obj.predicted_mean.clip(lower=0)
        fc_ci     = fc_obj.conf_int(alpha=0.2)

        # Build output dataframe
        train_preds = pd.DataFrame({"ds": ts.index, "yhat": y_pred_train.values})
        future_df   = pd.DataFrame({
            "ds":         fc_mean.index,
            "yhat":       fc_mean.values,
            "yhat_lower": fc_ci.iloc[:,0].clip(lower=0).values,
            "yhat_upper": fc_ci.iloc[:,1].values,
        })
        all_preds = pd.concat([train_preds, future_df], ignore_index=True)
        return all_preds, metrics
    except Exception as e:
        return None, None


# ── Ensemble Auto-Selection ───────────────────────────────────
def ensemble_forecast(train_df: pd.DataFrame, periods: int):
    """
    Jalankan ketiga model, evaluasi MAPE, buat weighted ensemble,
    dan pilih model terbaik otomatis.
    """
    results = {}

    # Prophet
    with st.spinner("🔵 Melatih Prophet..."):
        try:
            fc_p, met_p, _ = run_prophet(train_df, periods)
            results["Prophet"] = {"fc": fc_p, "metrics": met_p}
        except Exception as e:
            st.warning(f"Prophet gagal: {e}")

    # XGBoost
    with st.spinner("🟢 Melatih XGBoost..."):
        try:
            fc_x, met_x = run_xgboost(train_df, periods)
            if fc_x is not None:
                results["XGBoost"] = {"fc": fc_x, "metrics": met_x}
        except Exception as e:
            st.warning(f"XGBoost gagal: {e}")

    # SARIMA
    with st.spinner("🟡 Melatih SARIMA..."):
        try:
            fc_s, met_s = run_sarima(train_df, periods)
            if fc_s is not None:
                results["SARIMA"] = {"fc": fc_s, "metrics": met_s}
        except Exception as e:
            st.warning(f"SARIMA gagal: {e}")

    if not results:
        return None, None, None, None

    # Tentukan model terbaik berdasarkan MAPE (atau RMSE jika MAPE nan)
    def score(m):
        mape = m["metrics"].get("MAPE", np.nan) if m["metrics"] else np.nan
        rmse = m["metrics"].get("RMSE", np.nan) if m["metrics"] else np.nan
        return mape if not np.isnan(mape) else rmse

    best_name = min(results, key=lambda k: score(results[k]))
    best_fc   = results[best_name]["fc"]

    # Weighted ensemble untuk future predictions
    future_dfs = []
    weights    = []
    for name, r in results.items():
        sc = score(r)
        if sc and not np.isnan(sc):
            weights.append(1.0 / (sc + 1e-6))
            fc = r["fc"]
            if "yhat" in fc.columns:
                future_dfs.append(fc[["ds","yhat"]].rename(columns={"yhat": f"yhat_{name}"}))

    ensemble_fc = None
    if len(future_dfs) > 1:
        merged = future_dfs[0]
        for df_m in future_dfs[1:]:
            merged = merged.merge(df_m, on="ds", how="inner")
        yhat_cols = [c for c in merged.columns if c.startswith("yhat_")]
        total_w   = sum(weights[:len(yhat_cols)])
        merged["yhat_ensemble"] = sum(
            merged[col] * w for col, w in zip(yhat_cols, weights[:len(yhat_cols)])
        ) / total_w
        ensemble_fc = merged

    return results, best_name, best_fc, ensemble_fc


# ══════════════════════════════════════════════════════════════
# HALAMAN UTAMA: ENSEMBLE FORECASTING (page_ml_upgraded)
# ══════════════════════════════════════════════════════════════

def page_ml_upgraded(df_filtered, filter_info):
    st.subheader("🤖 Ensemble AI Forecasting (Prophet + XGBoost + SARIMA)")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong.")
        return
    if "tanggal_kunjungan" not in df_filtered.columns:
        st.error("❌ Kolom 'tanggal_kunjungan' tidak ditemukan.")
        return

    # ── Penjelasan arsitektur ──────────────────────────────────
    with st.expander("ℹ️ Tentang Arsitektur Ensemble Model", expanded=False):
        st.markdown("""
        | Model | Keunggulan | Kelemahan |
        |---|---|---|
        | **Prophet** | Tren + seasonality otomatis, robust terhadap missing data | Tidak memodelkan fitur eksogen secara eksplisit |
        | **XGBoost** | Feature engineering fleksibel (lag, musim, rolling) | Butuh data cukup untuk konvergensi |
        | **SARIMA** | Kuat untuk pola autoregresif musiman | Sensitif terhadap stasioneritas, lambat untuk data panjang |
        | **Ensemble** | Gabungan bobot berdasarkan MAPE terbaik → akurasi tertinggi | Lebih kompleks, waktu training lebih lama |

        **Auto-Selection:** Model dengan MAPE terendah dipilih sebagai *best single model*.
        **Weighted Ensemble:** Bobot proporsional terhadap 1/MAPE masing-masing model.
        """)

    df_ml = df_filtered.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_ml["tanggal_kunjungan"]):
        df_ml["tanggal_kunjungan"] = pd.to_datetime(df_ml["tanggal_kunjungan"], errors="coerce")
    df_ml = df_ml.dropna(subset=["tanggal_kunjungan"])
    if len(df_ml) == 0:
        st.warning("⚠️ Data tanggal tidak valid.")
        return

    max_date = df_ml["tanggal_kunjungan"].max().date()

    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1])
    with col1:
        fokus = st.radio("Analisis:", ["Diagnosa Penyakit", "Poli / Unit"], horizontal=True)
    kolom_fokus = "diagnosa" if fokus == "Diagnosa Penyakit" else "poli"
    with col2:
        if kolom_fokus not in df_ml.columns:
            st.error(f"Kolom '{kolom_fokus}' tidak ada.")
            return
        top_items   = df_ml[kolom_fokus].value_counts().head(30)
        pilihan_item = st.selectbox(f"Pilih {fokus}:", top_items.index.tolist())
    with col3:
        target_date = st.date_input(
            "Prediksi Sampai:",
            value=max_date + pd.Timedelta(days=60),
            min_value=max_date + pd.Timedelta(days=7),
            max_value=max_date + pd.Timedelta(days=365),
        )
    with col4:
        freq_label  = st.selectbox("Frekuensi:", ["Mingguan","Bulanan"])
        freq_map    = {"Mingguan": "W-MON", "Bulanan": "MS"}
        freq        = freq_map[freq_label]

    # ── Siapkan data time-series ──────────────────────────────
    df_item = df_ml[df_ml[kolom_fokus] == pilihan_item].copy()
    if len(df_item) < 15:
        st.error("❌ Data terlalu sedikit (< 15 baris) untuk ensemble forecasting.")
        return

    grouper   = pd.Grouper(key="tanggal_kunjungan", freq=freq)
    weekly    = df_item.groupby(grouper).size().reset_index(name="y")
    weekly    = weekly.rename(columns={"tanggal_kunjungan": "ds"})
    weekly    = weekly[weekly["y"] > 0]

    if len(weekly) < 8:
        st.error("❌ Data agregat terlalu sedikit setelah resampling.")
        return

    # Hitung periods
    last_date_ts = weekly["ds"].max()
    target_dt    = pd.to_datetime(target_date)
    if freq == "W-MON":
        periods = max(1, int((target_dt - last_date_ts).days // 7))
    else:
        periods = max(1, int((target_dt.year - last_date_ts.year) * 12 +
                             (target_dt.month - last_date_ts.month)))

    st.info(f"📊 Data: **{len(weekly)}** titik · Prediksi: **{periods}** {'minggu' if freq=='W-MON' else 'bulan'} ke depan")

    # ── Jalankan Ensemble ─────────────────────────────────────
    if st.button("🚀 Jalankan Ensemble Forecasting", type="primary"):
        results, best_name, best_fc, ensemble_fc = ensemble_forecast(weekly, periods)

        if not results:
            st.error("Semua model gagal. Coba dengan data lebih panjang.")
            return

        st.success(f"✅ Selesai! Model terbaik: **{best_name}**")

        # ── Tabel Evaluasi Akurasi ────────────────────────────
        st.markdown("### 📊 Evaluasi Akurasi Model")
        st.markdown("""
        > **MAE** = rata-rata error absolut · **RMSE** = root mean square error · **MAPE** = error persentase rata-rata
        > MAPE < 10% = Sangat Baik · 10–20% = Baik · > 20% = Perlu Perbaikan
        """)
        eval_rows = []
        for name, r in results.items():
            if r["metrics"]:
                row = {"Model": name, **r["metrics"]}
                row["Status"] = "🏆 Terbaik" if name == best_name else ""
                eval_rows.append(row)

        eval_df = pd.DataFrame(eval_rows)
        st.dataframe(
            eval_df.style.highlight_min(subset=["MAE","RMSE","MAPE"], color="#d1fae5"),
            use_container_width=True,
            hide_index=True,
        )

        # ── Grafik Perbandingan Model ─────────────────────────
        st.markdown(f"### 📈 Perbandingan Prediksi: **{pilihan_item}**")
        fig = go.Figure()

        # Aktual
        fig.add_trace(go.Scatter(
            x=weekly["ds"], y=weekly["y"],
            mode="lines+markers", name="Aktual",
            line=dict(color="#1d4ed8", width=2),
        ))

        colors = {"Prophet": "#ef4444", "XGBoost": "#16a34a", "SARIMA": "#d97706"}
        for name, r in results.items():
            fc = r["fc"]
            fc_future = fc[fc["ds"] > last_date_ts]
            if "yhat" in fc.columns:
                fig.add_trace(go.Scatter(
                    x=fc_future["ds"], y=fc_future["yhat"].clip(lower=0),
                    mode="lines", name=f"{name}",
                    line=dict(color=colors.get(name, "#6b7280"), width=2, dash="dot"),
                ))

        # Ensemble
        if ensemble_fc is not None:
            ens_future = ensemble_fc[ensemble_fc["ds"] > last_date_ts]
            if not ens_future.empty:
                fig.add_trace(go.Scatter(
                    x=ens_future["ds"], y=ens_future["yhat_ensemble"].clip(lower=0),
                    mode="lines+markers", name="⭐ Ensemble",
                    line=dict(color="#7c3aed", width=3),
                ))

        # Best model confidence interval (Prophet saja yang punya CI lengkap)
        if "Prophet" in results and best_name == "Prophet":
            fc_p = results["Prophet"]["fc"]
            fc_p_future = fc_p[fc_p["ds"] > last_date_ts]
            if {"yhat_upper","yhat_lower"}.issubset(fc_p_future.columns):
                fig.add_trace(go.Scatter(
                    x=fc_p_future["ds"].tolist() + fc_p_future["ds"].tolist()[::-1],
                    y=fc_p_future["yhat_upper"].tolist() + fc_p_future["yhat_lower"].tolist()[::-1],
                    fill="toself", fillcolor="rgba(239,68,68,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="CI Prophet 80%", hoverinfo="skip",
                ))

        fig.update_layout(
            xaxis_title="Periode", yaxis_title="Jumlah Kunjungan",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Ringkasan Estimasi ────────────────────────────────
        st.markdown(f"### 📢 Kesimpulan Estimasi Hingga {target_dt.strftime('%d %B %Y')}")

        # Gunakan ensemble atau best model
        use_fc = None
        use_label = ""
        if ensemble_fc is not None and "yhat_ensemble" in ensemble_fc.columns:
            use_fc    = ensemble_fc[ensemble_fc["ds"] > last_date_ts][["ds","yhat_ensemble"]].rename(columns={"yhat_ensemble":"yhat"})
            use_label = "Ensemble (berbobot)"
        elif best_fc is not None and "yhat" in best_fc.columns:
            use_fc    = best_fc[best_fc["ds"] > last_date_ts][["ds","yhat"]]
            use_label = best_name

        if use_fc is not None and not use_fc.empty:
            use_fc["yhat"] = use_fc["yhat"].clip(lower=0)
            total_est      = int(round(use_fc["yhat"].sum()))
            akhir_row      = use_fc.iloc[-1]
            tgl_akhir      = akhir_row["ds"].strftime("%d %B %Y")
            est_akhir      = int(round(akhir_row["yhat"]))

            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                st.markdown(f"""
                <div class="accuracy-box">
                <div class="highlight-estimasi">
                📆 Hingga <b>{tgl_akhir}</b>, model <b>{use_label}</b> memperkirakan 
                total akumulasi <b>{total_est:,} kunjungan/kasus</b> untuk <b>{pilihan_item}</b>.<br><br>
                Pada periode terakhir, diperkirakan <b>{est_akhir} kunjungan</b> baru.
                </div></div>
                """.replace(",","."), unsafe_allow_html=True)
            with col_b:
                st.metric("Total Estimasi Akumulasi", f"{total_est:,}".replace(",","."))
            with col_c:
                st.metric("Estimasi Periode Terakhir", f"{est_akhir}")

            # Download
            dl_df = use_fc.copy()
            dl_df.columns = ["Periode","Prediksi_Jumlah"]
            st.download_button(
                "📥 Download Prediksi (Excel)",
                convert_df_to_excel(dl_df),
                f"prediksi_{pilihan_item.lower().replace(' ','_')}.xlsx",
            )


# ══════════════════════════════════════════════════════════════
# HALAMAN: DETEKSI PENYAKIT DOMINAN PER MUSIM
# ══════════════════════════════════════════════════════════════

def page_disease_seasonality(df_filtered, filter_info):
    st.subheader("🌧️ Deteksi Penyakit Dominan per Musim / Bulan")
    show_active_filters(filter_info)

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("⚠️ Data kosong.")
        return
    if not {"tanggal_kunjungan","diagnosa"}.issubset(df_filtered.columns):
        st.error("❌ Butuh kolom 'tanggal_kunjungan' dan 'diagnosa'.")
        return

    df = df_filtered.copy()
    df["bulan"]       = df["tanggal_kunjungan"].dt.month
    df["nama_bulan"]  = df["tanggal_kunjungan"].dt.strftime("%b")
    df["musim"]       = df["bulan"].apply(lambda m: "🌧️ Hujan (Nov–Apr)" if m in [11,12,1,2,3,4] else "☀️ Kemarau (Mei–Okt)")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Heatmap Bulanan", "Penyakit per Musim", "Tren Penyakit Spesifik"])

    with tab1:
        st.markdown("#### 🗓️ Heatmap Frekuensi Penyakit per Bulan")
        top_n = st.slider("Jumlah penyakit ditampilkan", 5, 20, 10, key="heat_n")
        top_diag = df["diagnosa"].value_counts().head(top_n).index.tolist()
        df_heat = df[df["diagnosa"].isin(top_diag)].copy()
        pivot = df_heat.groupby(["diagnosa","bulan"]).size().unstack(fill_value=0)
        pivot.columns = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][:len(pivot.columns)]

        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Jumlah Kasus per Penyakit per Bulan",
            labels={"color":"Jumlah Kasus"},
        )
        fig_heat.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.download_button("📥 Download Heatmap Data", convert_df_to_excel(pivot.reset_index()), "heatmap_penyakit.xlsx")

    with tab2:
        st.markdown("#### 🌦️ Top Penyakit per Musim")
        col_a, col_b = st.columns(2)
        for musim_label, col_ui in zip(["🌧️ Hujan (Nov–Apr)", "☀️ Kemarau (Mei–Okt)"], [col_a, col_b]):
            df_musim = df[df["musim"] == musim_label]
            if df_musim.empty:
                col_ui.info(f"Tidak ada data untuk {musim_label}")
                continue
            top_diag_musim = df_musim["diagnosa"].value_counts().head(10).reset_index()
            top_diag_musim.columns = ["Diagnosa","Kasus"]
            fig = px.bar(
                top_diag_musim, x="Kasus", y="Diagnosa",
                orientation="h", title=musim_label,
                color="Kasus", color_continuous_scale="Blues",
                text="Kasus",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, height=350, yaxis=dict(categoryorder="total ascending"))
            col_ui.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📊 Perbandingan Intensitas Penyakit: Hujan vs Kemarau")
        df_comp = df.groupby(["musim","diagnosa"]).size().unstack(fill_value=0).T
        if df_comp.shape[1] == 2:
            df_comp.columns = ["Hujan","Kemarau"]
            df_comp["Selisih"] = df_comp["Hujan"] - df_comp["Kemarau"]
            df_comp["Dominan_Musim"] = df_comp["Selisih"].apply(
                lambda x: "🌧️ Hujan" if x > 0 else "☀️ Kemarau"
            )
            df_comp_top = df_comp.sort_values("Selisih", key=abs, ascending=False).head(15)
            st.dataframe(df_comp_top.reset_index().rename(columns={"diagnosa":"Diagnosa"}), use_container_width=True, hide_index=True)
            
        st.markdown("#### 📋 Rincian Lengkap Kasus per Musim")
        df_rincian = df.groupby(["diagnosa", "musim"]).size().reset_index(name="Jumlah Kasus")
        df_pivot = df_rincian.pivot(index="diagnosa", columns="musim", values="Jumlah Kasus").fillna(0).astype(int)
        df_pivot["Total Kasus"] = df_pivot.sum(axis=1)
        df_pivot = df_pivot.sort_values("Total Kasus", ascending=False).reset_index()
        st.dataframe(df_pivot, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### 📈 Tren Bulanan Penyakit Tertentu")
        top_all = df["diagnosa"].value_counts().head(20).index.tolist()
        penyakit_pilih = st.multiselect("Pilih penyakit untuk dibandingkan:", top_all, default=top_all[:3])
        if penyakit_pilih:
            df_trend = df[df["diagnosa"].isin(penyakit_pilih)].groupby(["bulan","diagnosa"]).size().reset_index(name="kasus")
            bulan_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
                         7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
            df_trend["nama_bulan"] = df_trend["bulan"].map(bulan_map)
            fig_trend = px.line(
                df_trend, x="nama_bulan", y="kasus",
                color="diagnosa", markers=True,
                title="Tren Kasus Bulanan per Penyakit",
                category_orders={"nama_bulan": list(bulan_map.values())},
            )
            st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN LAINNYA
# ══════════════════════════════════════════════════════════════

def page_overview(df_filtered, filter_info):
    st.subheader("📌 Ringkasan Umum")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0:
        st.warning("Tidak ada data.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Kunjungan", len(df_filtered))
    if "no_rm"    in df_filtered.columns: col2.metric("Pasien Unik", df_filtered["no_rm"].nunique())
    if "poli"     in df_filtered.columns: col3.metric("Poli Aktif",  df_filtered["poli"].nunique())
    if "diagnosa" in df_filtered.columns: col4.metric("Diagnosa",    df_filtered["diagnosa"].nunique())
    st.markdown("---")
    if "tanggal_kunjungan" in df_filtered.columns:
        st.markdown("### 📈 Tren Kunjungan")
        trend = df_filtered.groupby(["tahun","bulan","nama_bulan"]).size().reset_index(name="count") \
                           .sort_values(["tahun","bulan"])
        trend["label"] = trend["nama_bulan"].astype(str) + "-" + trend["tahun"].astype(str)
        st.line_chart(trend.set_index("label")["count"])
        st.download_button("📥 Download Tren", convert_df_to_excel(trend), "tren_kunjungan.xlsx")
    if "poli" in df_filtered.columns:
        st.markdown("### 🏥 Distribusi Poli")
        df_poli = df_filtered["poli"].value_counts().reset_index()
        df_poli.columns = ["Poli","Jumlah"]
        st.bar_chart(df_poli.set_index("Poli"))


def page_kunjungan(df_filtered, filter_info):
    st.subheader("👥 Analisis Kunjungan")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    col1, col2 = st.columns(2)
    if "jenis_kelamin" in df_filtered.columns:
        df_jk = df_filtered["jenis_kelamin"].value_counts().reset_index()
        df_jk.columns = ["Jenis Kelamin","Jumlah"]
        col1.markdown("#### Jenis Kelamin")
        col1.bar_chart(df_jk.set_index("Jenis Kelamin"))
    if "kelompok_umur" in df_filtered.columns:
        df_umur = df_filtered["kelompok_umur"].value_counts().sort_index().reset_index()
        df_umur.columns = ["Kelompok Umur","Jumlah"]
        col2.markdown("#### Kelompok Umur")
        col2.bar_chart(df_umur.set_index("Kelompok Umur"))


def page_penyakit(df_filtered, filter_info):
    st.subheader("🦠 Analisis Penyakit")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    if "diagnosa" not in df_filtered.columns:
        st.error("❌ Kolom 'diagnosa' tidak ditemukan.")
        return
        
    col_s, col_u, col_t = st.columns([2,2,1])
    with col_s: top_n = st.slider("Jumlah diagnosa", 5, 30, 10)
    with col_u:
        urutan = st.radio("Urut:", ["⬇️ Terbanyak","⬆️ Tersedikit"], horizontal=True)
    with col_t:
        orientasi = st.selectbox("Orientasi", ["Horizontal","Vertikal"])
        
    ascending = urutan.startswith("⬆️")
    df_diag = df_filtered["diagnosa"].value_counts().head(top_n).reset_index()
    df_diag.columns = ["Diagnosa","Jumlah Kasus"]
    df_diag = df_diag.sort_values("Jumlah Kasus", ascending=ascending).reset_index(drop=True)
    
    chart_h = max(350, top_n * 38)
    if orientasi == "Horizontal":
        fig = px.bar(df_diag, x="Jumlah Kasus", y="Diagnosa", orientation="h",
                     text="Jumlah Kasus", color="Jumlah Kasus", color_continuous_scale="Blues")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(categoryorder="total ascending" if ascending else "total descending"),
                          coloraxis_showscale=False, height=chart_h)
    else:
        fig = px.bar(df_diag, x="Diagnosa", y="Jumlah Kasus",
                     text="Jumlah Kasus", color="Jumlah Kasus", color_continuous_scale="Blues")
        fig.update_layout(xaxis=dict(tickangle=-35), coloraxis_showscale=False, height=chart_h)
        
    # --- 1. TAMPILKAN GRAFIK ---
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 2. TAMBAHAN: TABEL RINCIAN ---
    st.markdown("### 📋 Rincian Data Penyakit")
    st.dataframe(df_diag, use_container_width=True, hide_index=True)
    
    st.download_button("📥 Download Data Penyakit", convert_df_to_excel(df_diag), "top_penyakit.xlsx")


def page_peta_persebaran(df_filtered, filter_info):
    st.subheader("🗺️ Peta Persebaran Penyakit")
    show_active_filters(filter_info)
    if df_filtered is None or len(df_filtered) == 0: return
    if not {"desa","diagnosa"}.issubset(df_filtered.columns):
        st.error("❌ Butuh kolom 'desa' dan 'diagnosa'.")
        return
        
    top_penyakit = df_filtered["diagnosa"].value_counts().head(20).index.tolist()
    pilihan = st.selectbox("Filter Peta (Berdasarkan Diagnosa):", ["-- Semua Top 10 --"] + top_penyakit)
    
    if pilihan != "-- Semua Top 10 --":
        df_map = df_filtered[df_filtered["diagnosa"] == pilihan].copy()
    else:
        df_map = df_filtered[df_filtered["diagnosa"].isin(df_filtered["diagnosa"].value_counts().head(10).index)]
        
    df_grouped = df_map.groupby(["desa","diagnosa"]).size().reset_index(name="jumlah_kasus")
    
    koordinat_desa = {
        "Donan":(-7.2131,111.6364),"Gapluk":(-7.2017,111.6617),
        "Kaliombo":(-7.235,111.6817),"Kuniran":(-7.2346,111.6510),
        "Ngrejeng":(-7.2260,111.7071),"Pelem":(-7.2394,111.7011),
        "Pojok":(-7.1892,111.6728),"Punggur":(-7.2048,111.6808),
        "Purwosari":(-7.1798,111.6608),"Sedahkidul":(-7.1973,111.6792),
        "Tinumpuk":(-7.2117,111.68),"Tlatah":(-7.2172,111.6975),
    }
    def get_koord(desa): return koordinat_desa.get(str(desa).strip().title(), (-7.1509,111.8817))
    
    df_grouped["latitude"]  = df_grouped["desa"].apply(lambda x: get_koord(x)[0])
    df_grouped["longitude"] = df_grouped["desa"].apply(lambda x: get_koord(x)[1])
    
    # --- RENDER PETA DENGAN CUSTOM_DATA ---
    fig = px.scatter_mapbox(
        df_grouped, lat="latitude", lon="longitude",
        color="diagnosa", size="jumlah_kasus",
        hover_name="desa", hover_data={"diagnosa":True,"jumlah_kasus":True},
        custom_data=["desa"], # WAJIB ADA UNTUK EVENT KLIK
        zoom=11.5, center={"lat":-7.218,"lon":111.675}, height=550,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    
    st.markdown("*(💡 Klik salah satu titik/lingkaran pada peta untuk melihat detail data dari desa tersebut)*")
    
    # --- TANGKAP EVENT KLIK ---
    map_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

    # Ekstrak nama desa jika diklik
    selected_desa = None
    if map_event and map_event.get("selection") and map_event["selection"].get("points"):
        selected_desa = map_event["selection"]["points"][0]["customdata"][0]

    # --- TAMPILAN BERDASARKAN EVENT KLIK (URUTAN BARU) ---
    if selected_desa:
        st.markdown("---")
        st.markdown(f"### 📍 Rincian Khusus Desa: **{selected_desa}**")
        
        # Ambil data mentah khusus untuk desa ini
        df_desa_detail = df_filtered[df_filtered["desa"].str.title() == selected_desa]
        
        # 1. Tampilkan Grafik (Full Width)
        st.markdown("#### 📊 Top Diagnosa")
        df_desa_diag = df_desa_detail["diagnosa"].value_counts().head(10).reset_index()
        df_desa_diag.columns = ["Diagnosa", "Jumlah Kasus"]
        fig_desa = px.bar(df_desa_diag, x="Jumlah Kasus", y="Diagnosa", orientation="h", color="Jumlah Kasus", color_continuous_scale="Viridis")
        fig_desa.update_layout(yaxis=dict(categoryorder="total ascending"), height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_desa, use_container_width=True)
        
        # 2. Tampilkan Tabel (Full Width di bawah grafik)
        st.markdown("#### 📋 Data Histori Kunjungan")
        kolom_tabel = [c for c in ["tanggal_kunjungan", "no_rm", "umur", "jenis_kelamin", "diagnosa", "poli"] if c in df_desa_detail.columns]
        st.dataframe(df_desa_detail[kolom_tabel], use_container_width=True, hide_index=True)
            
    else:
        # Tampilan default jika belum ada titik yang diklik
        st.markdown("### 📋 Tabel Akumulasi (Seluruh Desa yang Tampil)")
        st.dataframe(df_grouped.sort_values("jumlah_kasus", ascending=False).drop(columns=["latitude", "longitude"]), use_container_width=True, hide_index=True)


def page_pembiayaan(df_filtered, filter_info):
    st.subheader("💳 Analisis Pembiayaan")
    if df_filtered is None or "pembiayaan" not in df_filtered.columns: return
    df_b = df_filtered["pembiayaan"].value_counts().reset_index()
    df_b.columns = ["Pembiayaan","Jumlah"]
    st.bar_chart(df_b.set_index("Pembiayaan"))
    st.download_button("📥 Download Pembiayaan", convert_df_to_excel(df_b), "pembiayaan.xlsx")


def page_data(df_filtered, filter_info):
    st.subheader("📄 Data & Unduhan")
    if df_filtered is None: return
    st.dataframe(df_filtered)
    st.download_button("💾 Download CSV", df_filtered.to_csv(index=False).encode("utf-8"), "data_puskesmas.csv", "text/csv")


def page_quality(df):
    st.subheader("🧹 Kualitas Data")
    if df is None: return
    st.write(f"Duplikasi: **{df.duplicated().sum()}** baris")
    st.markdown("**Missing Values:**")
    st.dataframe(df.isna().sum().to_frame("Missing Count"), use_container_width=True)


def page_ai_assistant(df_filtered, filter_info, is_genai):
    st.subheader("🤖 Asisten AI (Gemini)")
    if df_filtered is None or df_filtered.empty:
        st.warning("Upload dan filter data terlebih dahulu.")
        return
    if not is_genai:
        st.error("❌ API Key Gemini belum diset.")
        return
    total = len(df_filtered)
    top_dx = ", ".join([f"{k}({v})" for k,v in df_filtered["diagnosa"].value_counts().head(5).items()]) \
             if "diagnosa" in df_filtered.columns else "-"
    top_pl = ", ".join([f"{k}({v})" for k,v in df_filtered["poli"].value_counts().head(3).items()]) \
             if "poli" in df_filtered.columns else "-"
    ctx = f"""[DATA REAL-TIME]
- Total kunjungan: {total}
- 5 penyakit terbanyak: {top_dx}
- 3 poli terpadat: {top_pl}"""
    user_q = st.text_area("Tanyakan strategi/analisis:", placeholder="Contoh: Program promkes apa yang paling mendesak?")
    if st.button("Kirim"):
        if not user_q.strip(): st.warning("Pertanyaan kosong."); return
        prompt = f"""Anda adalah Analis Kesehatan Masyarakat di UPT Puskesmas Purwosari (Bojonegoro).
{ctx}
Jawab pertanyaan berikut secara spesifik, berbasis data, dan terstruktur:
{user_q}"""
        with st.spinner("AI menganalisis..."):
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                resp  = model.generate_content(prompt)
                st.markdown("### 📊 Analisis AI:")
                st.markdown(resp.text)
            except Exception as e:
                st.error(f"Gagal: {e}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    is_genai = get_gemini_client()
    df_filtered, filter_info = apply_filters(None)

    pages = {
        "📌 Ringkasan":                page_overview,
        "👥 Analisis Kunjungan":       page_kunjungan,
        "🦠 Analisis Penyakit":        page_penyakit,
        "🌧️ Penyakit per Musim":      page_disease_seasonality,
        "🗺️ Peta Persebaran":         page_peta_persebaran,
        "🤖 Ensemble Forecasting":     page_ml_upgraded,
        "💳 Pembiayaan":               page_pembiayaan,
        "🧹 Kualitas Data":            page_quality,
        "📄 Data & Unduhan":           page_data,
        "💬 Asisten AI":               None,
    }

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🗂️ Navigasi")
        page_sel = st.radio("Pilih halaman:", list(pages.keys()), label_visibility="collapsed")

    if df_filtered is None:
        st.info("👈 Upload file data di sidebar untuk memulai.")
        return

    if page_sel == "💬 Asisten AI":
        page_ai_assistant(df_filtered, filter_info, is_genai)
    elif page_sel == "🧹 Kualitas Data":
        page_quality(df_filtered)
    else:
        fn = pages[page_sel]
        if fn: fn(df_filtered, filter_info)

if __name__ == "__main__":
    main()
