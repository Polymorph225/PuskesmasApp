import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Sistem Informasi Puskesmas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stMetric { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .section-header { font-size: 1.3rem; font-weight: 700; color: #1a365d; margin: 1.5rem 0 0.5rem; border-left: 4px solid #3182ce; padding-left: 10px; }
    .table-title { font-size: 1rem; font-weight: 600; color: #2d3748; margin: 1rem 0 0.3rem; }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)


# ── Sample Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)

    desa_list = [
        {"nama": "Desa Sukamaju",  "lat": -7.2575, "lon": 112.7521, "kecamatan": "Kec. Wonokromo"},
        {"nama": "Desa Harapan",   "lat": -7.2650, "lon": 112.7450, "kecamatan": "Kec. Wonokromo"},
        {"nama": "Desa Mekarjaya", "lat": -7.2700, "lon": 112.7600, "kecamatan": "Kec. Gayungan"},
        {"nama": "Desa Sejahtera", "lat": -7.2500, "lon": 112.7480, "kecamatan": "Kec. Gayungan"},
        {"nama": "Desa Bahagia",   "lat": -7.2620, "lon": 112.7350, "kecamatan": "Kec. Jambangan"},
        {"nama": "Desa Mandiri",   "lat": -7.2450, "lon": 112.7580, "kecamatan": "Kec. Jambangan"},
        {"nama": "Desa Berdikari", "lat": -7.2730, "lon": 112.7490, "kecamatan": "Kec. Karang Pilang"},
        {"nama": "Desa Merdeka",   "lat": -7.2550, "lon": 112.7420, "kecamatan": "Kec. Karang Pilang"},
    ]

    penyakit_list = [
        "ISPA", "Diare", "Hipertensi", "Diabetes", "DBD",
        "Malaria", "TBC", "Scabies", "Pneumonia", "Disentri"
    ]

    bulan_list = pd.date_range("2022-01-01", "2024-12-01", freq="MS")

    records = []
    for desa in desa_list:
        for bulan in bulan_list:
            for penyakit in penyakit_list:
                jumlah = max(0, int(
                    np.random.poisson(lam=20 if penyakit in ["ISPA", "Diare", "Hipertensi"] else 8)
                    + np.random.normal(0, 3)
                ))
                records.append({
                    "desa": desa["nama"],
                    "lat": desa["lat"],
                    "lon": desa["lon"],
                    "kecamatan": desa["kecamatan"],
                    "bulan": bulan,
                    "tahun": bulan.year,
                    "bulan_num": bulan.month,
                    "penyakit": penyakit,
                    "jumlah": jumlah
                })

    df = pd.DataFrame(records)

    def get_musim(m):
        if m in [11, 12, 1, 2, 3]:
            return "Hujan"
        elif m in [6, 7, 8, 9]:
            return "Kemarau"
        elif m in [4, 5]:
            return "Peralihan Hujan-Kemarau"
        else:
            return "Peralihan Kemarau-Hujan"

    df["musim"] = df["bulan_num"].apply(get_musim)
    df["nama_bulan"] = df["bulan"].dt.strftime("%b %Y")
    return df, desa_list


df, desa_list = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Sistem Informasi Puskesmas")
    st.divider()
    page = st.radio(
        "Navigasi",
        ["Dashboard", "Analisis Penyakit", "Penyakit per Musim",
         "Peta Persebaran", "Prediksi SARIMA"],
        label_visibility="collapsed"
    )
    st.divider()
    tahun_filter = st.multiselect("Filter Tahun", [2022, 2023, 2024], default=[2022, 2023, 2024])
    penyakit_filter = st.multiselect(
        "Filter Penyakit", sorted(df["penyakit"].unique()),
        default=list(df["penyakit"].unique())
    )

df_filtered = df[df["tahun"].isin(tahun_filter) & df["penyakit"].isin(penyakit_filter)]

URUTAN_MUSIM = ["Hujan", "Peralihan Hujan-Kemarau", "Kemarau", "Peralihan Kemarau-Hujan"]
WARNA_MUSIM = {
    "Hujan": "#3182ce", "Peralihan Hujan-Kemarau": "#48bb78",
    "Kemarau": "#ed8936", "Peralihan Kemarau-Hujan": "#9f7aea"
}


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Dashboard Sistem Informasi Puskesmas")
    st.caption("Ringkasan data kunjungan dan penyakit wilayah kerja puskesmas")

    total_kasus = df_filtered["jumlah"].sum()
    total_desa = df_filtered["desa"].nunique()
    top_penyakit = df_filtered.groupby("penyakit")["jumlah"].sum().idxmax()
    kasus_bulan = df_filtered.groupby("bulan")["jumlah"].sum().iloc[-1] if not df_filtered.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Kasus", f"{total_kasus:,}")
    c2.metric("Jumlah Desa", total_desa)
    c3.metric("Penyakit Tertinggi", top_penyakit)
    c4.metric("Kasus Bulan Terakhir", f"{kasus_bulan:,}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        trend = df_filtered.groupby("bulan")["jumlah"].sum().reset_index()
        fig = px.line(trend, x="bulan", y="jumlah", title="Tren Total Kasus per Bulan",
                      markers=True, color_discrete_sequence=["#3182ce"])
        fig.update_layout(xaxis_title="Bulan", yaxis_title="Jumlah Kasus", height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top10 = df_filtered.groupby("penyakit")["jumlah"].sum().nlargest(10).reset_index()
        fig2 = px.bar(top10, x="jumlah", y="penyakit", orientation="h",
                      title="Top 10 Penyakit", color="jumlah", color_continuous_scale="Blues")
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=320)
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALISIS PENYAKIT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Analisis Penyakit":
    st.title("Analisis Penyakit")

    # Grafik 1
    st.markdown('<p class="section-header">Total Kasus per Penyakit</p>', unsafe_allow_html=True)
    per_penyakit = (df_filtered.groupby("penyakit")["jumlah"].sum()
                    .reset_index().sort_values("jumlah", ascending=False))
    fig1 = px.bar(per_penyakit, x="penyakit", y="jumlah", color="jumlah",
                  color_continuous_scale="Teal", text="jumlah")
    fig1.update_traces(textposition="outside")
    fig1.update_layout(xaxis_title="Penyakit", yaxis_title="Total Kasus", height=380, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<p class="table-title">Tabel Rincian: Total Kasus per Penyakit</p>', unsafe_allow_html=True)
    tbl1 = per_penyakit.copy()
    tbl1["Persentase (%)"] = (tbl1["jumlah"] / tbl1["jumlah"].sum() * 100).round(2)
    tbl1.insert(0, "Peringkat", range(1, len(tbl1) + 1))
    tbl1 = tbl1.rename(columns={"penyakit": "Penyakit", "jumlah": "Total Kasus"})
    st.dataframe(tbl1, use_container_width=True, hide_index=True)

    st.divider()

    # Grafik 2
    st.markdown('<p class="section-header">Tren Bulanan per Penyakit</p>', unsafe_allow_html=True)
    penyakit_sel = st.multiselect(
        "Pilih Penyakit untuk Tren:", sorted(df_filtered["penyakit"].unique()),
        default=list(df_filtered["penyakit"].unique())[:3]
    )
    if penyakit_sel:
        tren_df = (df_filtered[df_filtered["penyakit"].isin(penyakit_sel)]
                   .groupby(["bulan", "penyakit"])["jumlah"].sum().reset_index())
        fig2 = px.line(tren_df, x="bulan", y="jumlah", color="penyakit",
                       markers=True, title="Tren Kasus Penyakit per Bulan")
        fig2.update_layout(xaxis_title="Bulan", yaxis_title="Jumlah Kasus", height=380)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="table-title">Tabel Rincian: Tren Bulanan per Penyakit</p>', unsafe_allow_html=True)
        tbl2 = tren_df.copy()
        tbl2["bulan"] = tbl2["bulan"].dt.strftime("%b %Y")
        tbl2 = tbl2.rename(columns={"bulan": "Bulan", "penyakit": "Penyakit", "jumlah": "Jumlah Kasus"})
        st.dataframe(tbl2.sort_values(["Penyakit", "Bulan"]).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    st.divider()

    # Grafik 3
    st.markdown('<p class="section-header">Distribusi Penyakit per Desa (Heatmap)</p>', unsafe_allow_html=True)
    heatmap_df = df_filtered.pivot_table(
        index="desa", columns="penyakit", values="jumlah", aggfunc="sum"
    ).fillna(0)
    fig3 = px.imshow(heatmap_df, aspect="auto", color_continuous_scale="YlOrRd",
                     title="Heatmap Penyakit per Desa", text_auto=True)
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="table-title">Tabel Rincian: Distribusi per Desa & Penyakit</p>', unsafe_allow_html=True)
    tbl3 = (df_filtered.groupby(["desa", "penyakit"])["jumlah"].sum().reset_index()
            .rename(columns={"desa": "Desa", "penyakit": "Penyakit", "jumlah": "Total Kasus"})
            .sort_values(["Desa", "Total Kasus"], ascending=[True, False]).reset_index(drop=True))
    st.dataframe(tbl3, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PENYAKIT PER MUSIM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Penyakit per Musim":
    st.title("Penyakit per Musim")
    st.caption("Analisis pola penyakit berdasarkan musim")

    # Grafik 1
    st.markdown('<p class="section-header">Total Kasus per Musim</p>', unsafe_allow_html=True)
    per_musim = (df_filtered.groupby("musim")["jumlah"].sum()
                 .reindex(URUTAN_MUSIM).reset_index())
    fig1 = px.bar(per_musim, x="musim", y="jumlah", color="musim",
                  color_discrete_map=WARNA_MUSIM, text="jumlah",
                  title="Total Kasus Berdasarkan Musim")
    fig1.update_traces(textposition="outside")
    fig1.update_layout(xaxis_title="Musim", yaxis_title="Total Kasus", height=380, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<p class="table-title">Tabel Rincian: Total Kasus per Musim</p>', unsafe_allow_html=True)
    tbl_m = per_musim.copy()
    tbl_m["Persentase (%)"] = (tbl_m["jumlah"] / tbl_m["jumlah"].sum() * 100).round(2)
    tbl_m = tbl_m.rename(columns={"musim": "Musim", "jumlah": "Total Kasus"})
    st.dataframe(tbl_m, use_container_width=True, hide_index=True)

    st.divider()

    # Grafik 2
    st.markdown('<p class="section-header">Distribusi Penyakit per Musim</p>', unsafe_allow_html=True)
    pm = df_filtered.groupby(["musim", "penyakit"])["jumlah"].sum().reset_index()
    pm["musim"] = pd.Categorical(pm["musim"], categories=URUTAN_MUSIM, ordered=True)
    pm = pm.sort_values("musim")
    fig2 = px.bar(pm, x="penyakit", y="jumlah", color="musim",
                  barmode="group", color_discrete_map=WARNA_MUSIM,
                  title="Distribusi Penyakit per Musim")
    fig2.update_layout(xaxis_title="Penyakit", yaxis_title="Jumlah Kasus", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="table-title">Tabel Rincian: Penyakit per Musim (Pivot)</p>', unsafe_allow_html=True)
    pivot_tbl2 = pm.rename(columns={"musim": "Musim", "penyakit": "Penyakit", "jumlah": "Jumlah Kasus"})
    pivot_tbl2 = (pivot_tbl2.pivot_table(index="Penyakit", columns="Musim",
                                         values="Jumlah Kasus", aggfunc="sum", fill_value=0)
                  .reset_index())
    pivot_tbl2["Total"] = pivot_tbl2.drop(columns="Penyakit").sum(axis=1)
    pivot_tbl2 = pivot_tbl2.sort_values("Total", ascending=False).reset_index(drop=True)
    st.dataframe(pivot_tbl2, use_container_width=True, hide_index=True)

    st.divider()

    # Grafik 3 Radar
    st.markdown('<p class="section-header">Radar Chart Intensitas Penyakit per Musim</p>', unsafe_allow_html=True)
    penyakit_radar = sorted(df_filtered["penyakit"].unique())
    fig3 = go.Figure()
    for musim in URUTAN_MUSIM:
        vals = [
            df_filtered[(df_filtered["musim"] == musim) & (df_filtered["penyakit"] == p)]["jumlah"].sum()
            for p in penyakit_radar
        ]
        vals.append(vals[0])
        fig3.add_trace(go.Scatterpolar(
            r=vals, theta=penyakit_radar + [penyakit_radar[0]],
            fill="toself", name=musim, line_color=WARNA_MUSIM[musim]
        ))
    fig3.update_layout(polar=dict(radialaxis=dict(visible=True)),
                       title="Radar Penyakit per Musim", height=420)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="table-title">Tabel Rincian: Statistik per Musim & Penyakit</p>', unsafe_allow_html=True)
    tbl3 = df_filtered.groupby(["musim", "penyakit"])["jumlah"].agg(
        Total="sum", Rata_Rata="mean", Maks="max", Min="min"
    ).reset_index()
    tbl3["Rata_Rata"] = tbl3["Rata_Rata"].round(1)
    tbl3.columns = ["Musim", "Penyakit", "Total Kasus", "Rata-rata", "Maks", "Min"]
    st.dataframe(tbl3, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PETA PERSEBARAN  — Plotly Mapbox (tanpa folium / streamlit-folium)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Peta Persebaran":
    st.title("Peta Persebaran Penyakit")
    st.caption(
        "Klik titik pada peta atau gunakan tombol/dropdown di bawah "
        "untuk melihat detail grafik dan tabel data per desa."
    )

    # session state
    if "desa_dipilih" not in st.session_state:
        st.session_state["desa_dipilih"] = None

    # Agregasi
    desa_totals = (
        df_filtered.groupby(["desa", "lat", "lon", "kecamatan"])["jumlah"]
        .sum().reset_index()
    )
    max_val = desa_totals["jumlah"].max()
    desa_totals["ukuran"] = (desa_totals["jumlah"] / max_val * 40 + 15).round(1)

    # Peta
    fig_map = px.scatter_mapbox(
        desa_totals,
        lat="lat", lon="lon",
        size="ukuran",
        color="jumlah",
        color_continuous_scale="Reds",
        hover_name="desa",
        hover_data={"kecamatan": True, "jumlah": True,
                    "lat": False, "lon": False, "ukuran": False},
        labels={"jumlah": "Total Kasus", "kecamatan": "Kecamatan"},
        zoom=13,
        center={"lat": desa_totals["lat"].mean(), "lon": desa_totals["lon"].mean()},
        height=500,
        size_max=50,
    )
    fig_map.add_trace(go.Scattermapbox(
        lat=desa_totals["lat"].tolist(),
        lon=desa_totals["lon"].tolist(),
        mode="text",
        text=desa_totals["desa"].str.replace("Desa ", "").tolist(),
        textfont=dict(size=11, color="#1a365d"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Total Kasus"),
    )

    col_map, col_leg = st.columns([4, 1])
    with col_map:
        click_event = st.plotly_chart(
            fig_map, use_container_width=True,
            on_select="rerun", key="peta_plotly"
        )
    with col_leg:
        st.markdown("#### Pilih Desa")
        for _, row in desa_totals.sort_values("jumlah", ascending=False).iterrows():
            if st.button(
                f"{row['desa']}  ({int(row['jumlah']):,})",
                key=f"btn_{row['desa']}", use_container_width=True
            ):
                st.session_state["desa_dipilih"] = row["desa"]
                st.rerun()

    # Deteksi klik peta
    if (click_event and isinstance(click_event, dict)
            and click_event.get("selection")
            and click_event["selection"].get("points")):
        pt = click_event["selection"]["points"][0]
        clat, clon = pt.get("lat"), pt.get("lon")
        if clat is not None and clon is not None:
            desa_totals["_dist"] = np.sqrt(
                (desa_totals["lat"] - clat) ** 2 + (desa_totals["lon"] - clon) ** 2
            )
            nearest = desa_totals.loc[desa_totals["_dist"].idxmin(), "desa"]
            if nearest != st.session_state.get("desa_dipilih"):
                st.session_state["desa_dipilih"] = nearest
                st.rerun()

    # Selector manual
    opts = ["-- Pilih desa --"] + sorted(desa_totals["desa"].tolist())
    idx = opts.index(st.session_state["desa_dipilih"]) if st.session_state["desa_dipilih"] in opts else 0
    sel = st.selectbox("Atau pilih dari daftar:", opts, index=idx, key="selectbox_desa")
    if sel != "-- Pilih desa --":
        st.session_state["desa_dipilih"] = sel

    desa_diklik = st.session_state["desa_dipilih"]

    # ── Detail desa ───────────────────────────────────────────────────────────
    if not desa_diklik:
        st.info("Klik titik pada peta, tekan tombol desa di sebelah kanan, atau pilih dari dropdown.")
    else:
        st.success(f"Menampilkan data untuk: **{desa_diklik}**")
        st.divider()
        st.markdown(f"## Detail Data: {desa_diklik}")

        df_desa = df_filtered[df_filtered["desa"] == desa_diklik]
        kecamatan = df_desa["kecamatan"].iloc[0] if not df_desa.empty else "-"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Kasus", f"{df_desa['jumlah'].sum():,}")
        m2.metric("Kecamatan", kecamatan)
        m3.metric("Penyakit Tertinggi",
                  df_desa.groupby("penyakit")["jumlah"].sum().idxmax() if not df_desa.empty else "-")
        m4.metric("Periode Data", f"{df_desa['tahun'].min()}-{df_desa['tahun'].max()}")

        c1, c2 = st.columns(2)
        with c1:
            per_p = (df_desa.groupby("penyakit")["jumlah"].sum()
                     .reset_index().sort_values("jumlah", ascending=True))
            fig_p = px.bar(per_p, x="jumlah", y="penyakit", orientation="h",
                           color="jumlah", color_continuous_scale="Reds",
                           title=f"Total Kasus per Penyakit - {desa_diklik}", text="jumlah")
            fig_p.update_traces(textposition="outside")
            fig_p.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            tren_d = df_desa.groupby("bulan")["jumlah"].sum().reset_index()
            fig_t = px.line(tren_d, x="bulan", y="jumlah", markers=True,
                            title=f"Tren Kasus Bulanan - {desa_diklik}",
                            color_discrete_sequence=["#e53e3e"])
            fig_t.update_layout(xaxis_title="Bulan", yaxis_title="Jumlah Kasus", height=380)
            st.plotly_chart(fig_t, use_container_width=True)

        per_musim_d = (df_desa.groupby("musim")["jumlah"].sum()
                       .reindex(URUTAN_MUSIM).reset_index())
        fig_m = px.pie(per_musim_d, names="musim", values="jumlah",
                       title=f"Distribusi per Musim - {desa_diklik}",
                       color_discrete_sequence=["#3182ce", "#48bb78", "#ed8936", "#9f7aea"])
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown(f'<p class="table-title">Tabel Rincian Data - {desa_diklik}</p>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Per Penyakit", "Per Bulan", "Per Musim"])

        with tab1:
            tbl_p = df_desa.groupby("penyakit")["jumlah"].agg(
                Total="sum", Rata_rata="mean", Maks="max", Min="min"
            ).reset_index().sort_values("Total", ascending=False)
            tbl_p["Rata_rata"] = tbl_p["Rata_rata"].round(1)
            tbl_p["% dari Total"] = (tbl_p["Total"] / tbl_p["Total"].sum() * 100).round(2)
            tbl_p.columns = ["Penyakit", "Total", "Rata-rata/Bulan", "Maks", "Min", "% dari Total"]
            st.dataframe(tbl_p, use_container_width=True, hide_index=True)

        with tab2:
            tbl_b = df_desa.groupby(["bulan", "penyakit"])["jumlah"].sum().reset_index()
            tbl_b["bulan"] = tbl_b["bulan"].dt.strftime("%b %Y")
            tbl_b.columns = ["Bulan", "Penyakit", "Jumlah Kasus"]
            st.dataframe(
                tbl_b.sort_values(["Bulan", "Jumlah Kasus"], ascending=[True, False]).reset_index(drop=True),
                use_container_width=True, hide_index=True
            )

        with tab3:
            tbl_ms = df_desa.groupby(["musim", "penyakit"])["jumlah"].sum().reset_index()
            tbl_ms.columns = ["Musim", "Penyakit", "Total Kasus"]
            st.dataframe(
                tbl_ms.sort_values(["Musim", "Total Kasus"], ascending=[True, False]).reset_index(drop=True),
                use_container_width=True, hide_index=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PREDIKSI SARIMA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Prediksi SARIMA":
    st.title("Prediksi SARIMA")
    st.caption("Prediksi jumlah kasus penyakit menggunakan model SARIMA")

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        SARIMA_AVAILABLE = True
    except ImportError:
        SARIMA_AVAILABLE = False
        st.warning("Package statsmodels tidak terinstall. Jalankan: pip install statsmodels")

    col1, col2 = st.columns(2)
    with col1:
        penyakit_sarima = st.selectbox("Pilih Penyakit:", sorted(df["penyakit"].unique()))
    with col2:
        desa_sarima = st.selectbox("Pilih Desa:", ["Semua Desa"] + [d["nama"] for d in desa_list])

    bulan_prediksi = st.slider("Jumlah Bulan Prediksi:", 3, 24, 6)

    if st.button("Jalankan Prediksi SARIMA", type="primary") and SARIMA_AVAILABLE:
        with st.spinner("Melatih model SARIMA..."):
            try:
                if desa_sarima == "Semua Desa":
                    ts_df = df[df["penyakit"] == penyakit_sarima].groupby("bulan")["jumlah"].sum()
                else:
                    ts_df = df[
                        (df["penyakit"] == penyakit_sarima) & (df["desa"] == desa_sarima)
                    ].groupby("bulan")["jumlah"].sum()

                ts_df = ts_df.asfreq("MS")
                # FIX pandas >= 2.0: ffill()/bfill() menggantikan fillna(method=...)
                ts_df = ts_df.ffill().bfill()

                if len(ts_df) < 12:
                    st.warning("Data terlalu sedikit untuk SARIMA. Minimal 12 bulan.")
                else:
                    model = SARIMAX(ts_df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False, enforce_invertibility=False)
                    result = model.fit(disp=False)

                    forecast = result.get_forecast(steps=bulan_prediksi)
                    pred_mean = forecast.predicted_mean
                    pred_ci = forecast.conf_int()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df.values,
                                             mode="lines+markers", name="Historis",
                                             line=dict(color="#3182ce")))
                    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values,
                                             mode="lines+markers", name="Prediksi",
                                             line=dict(color="#e53e3e", dash="dash")))
                    fig.add_trace(go.Scatter(
                        x=list(pred_ci.index) + list(pred_ci.index[::-1]),
                        y=list(pred_ci.iloc[:, 1]) + list(pred_ci.iloc[:, 0][::-1]),
                        fill="toself", fillcolor="rgba(229,62,62,0.15)",
                        line=dict(color="rgba(0,0,0,0)"), name="Interval Kepercayaan 95%"
                    ))
                    fig.update_layout(
                        title=f"Prediksi SARIMA - {penyakit_sarima} ({desa_sarima})",
                        xaxis_title="Bulan", yaxis_title="Jumlah Kasus", height=420
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown('<p class="table-title">Tabel Hasil Prediksi</p>', unsafe_allow_html=True)
                    pred_tbl = pd.DataFrame({
                        "Bulan": pred_mean.index.strftime("%b %Y"),
                        "Prediksi Kasus": pred_mean.values.round(1),
                        "Batas Bawah (95%)": pred_ci.iloc[:, 0].values.round(1),
                        "Batas Atas (95%)": pred_ci.iloc[:, 1].values.round(1),
                    })
                    st.dataframe(pred_tbl, use_container_width=True, hide_index=True)

                    with st.expander("Statistik Model SARIMA"):
                        ca, cb, cc = st.columns(3)
                        ca.metric("AIC", f"{result.aic:.2f}")
                        cb.metric("BIC", f"{result.bic:.2f}")
                        cc.metric("Log-Likelihood", f"{result.llf:.2f}")

            except Exception as e:
                st.error(f"SARIMA gagal: {str(e)}")
                st.info("Coba kurangi kompleksitas order SARIMA atau periksa data Anda.")

    elif not SARIMA_AVAILABLE:
        st.info("Install statsmodels terlebih dahulu untuk menggunakan fitur prediksi.")
    else:
        st.info("Pilih parameter dan klik tombol Jalankan Prediksi SARIMA untuk memulai.")
