"""
╔══════════════════════════════════════════════════════════╗
║        EUROMILLIONS · ANALYSE & PRONOSTIC               ║
║        Biostatistique & Clustering des tirages          ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from stats_analysis import (
    load_and_clean, compute_frequencies, compute_retards,
    compute_pairs, compute_triplets, compute_gaps,
    chi_square_test, sliding_window_trend, sum_distribution,
    parity_analysis, decade_analysis,
)
from predictor import (
    build_draw_features, fit_kmeans, optimal_k,
    cluster_profiles, compute_heat_scores,
    generate_predictions, markov_transition, consecutive_analysis,
)

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="EuroMillions · Analyse & Pronostic",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS PERSONNALISÉ ─────────────────────────────────────
st.markdown("""
<style>
/* Fond principal */
.main { background-color: #0A0E1A; }

/* Boules jaunes */
.ball {
    display: inline-flex; align-items: center; justify-content: center;
    width: 46px; height: 46px; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #FFE066, #E6A800);
    color: #1A1A00; font-weight: 900; font-size: 15px;
    margin: 4px; box-shadow: 0 3px 10px rgba(255,215,0,0.5);
    border: 2px solid #FFD700;
}
/* Étoiles bleues */
.star {
    display: inline-flex; align-items: center; justify-content: center;
    width: 46px; height: 46px; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #66C2FF, #005FA3);
    color: #fff; font-weight: 900; font-size: 15px;
    margin: 4px; box-shadow: 0 3px 10px rgba(0,150,255,0.5);
    border: 2px solid #0080CC;
}
/* Cards */
.metric-card {
    background: #141826; border-radius: 12px;
    padding: 16px 20px; margin: 8px 0;
    border-left: 4px solid #FFD700;
}
.card-title { color: #AAB0C6; font-size: 12px; text-transform: uppercase; }
.card-value { color: #FFD700; font-size: 28px; font-weight: 800; }

/* Grille de pronostic */
.pronostic-card {
    background: linear-gradient(135deg, #141826, #1E2438);
    border-radius: 16px; padding: 20px;
    border: 1px solid #2A3050; margin: 10px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.rank-badge {
    background: #FFD700; color: #000; border-radius: 8px;
    padding: 2px 10px; font-size: 12px; font-weight: 700;
}

/* Séparateur */
hr { border-color: #2A3050; }

/* Heatmap labels */
.hot  { color: #FF6B6B; font-weight: 700; }
.cold { color: #4FC3F7; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎰 EuroMillions Analyser")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Charger l'historique (Excel .xlsx)",
        type=["xlsx", "xls"],
        help="Fichier Excel avec colonnes de boules (1-50) et étoiles (1-12)"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres d'analyse")

    n_clusters = st.slider("Nombre de clusters (K-Means)", 3, 15, 8,
                           help="Nombre de groupes de tirages similaires")
    recent_window = st.slider("Fenêtre récente (tirages)", 10, 100, 30,
                              help="Taille de la fenêtre pour la tendance récente")
    n_grilles = st.slider("Nombre de grilles à générer", 1, 10, 5)

    strategy = st.selectbox(
        "Stratégie de pronostic",
        ["balanced", "hot", "cold"],
        format_func=lambda x: {
            "balanced": "⚖️ Équilibrée (recommandée)",
            "hot": "🔥 Boules chaudes",
            "cold": "❄️ Boules froides",
        }[x]
    )

    w_freq   = st.slider("Poids Fréquence",   0.0, 1.0, 0.35, 0.05)
    w_retard = st.slider("Poids Retard",      0.0, 1.0, 0.30, 0.05)
    w_trend  = st.slider("Poids Tendance",    0.0, 1.0, 0.20, 0.05)
    w_cluster= st.slider("Poids Cluster",     0.0, 1.0, 0.15, 0.05)

    st.markdown("---")
    st.caption("ℹ️ Le hasard reste fondamental : ces analyses sont statistiques, "
               "non déterministes.")


# ═══════════════════════════════════════════════════════
#  HELPERS UI
# ═══════════════════════════════════════════════════════

def balls_html(numbers: list, type_="ball") -> str:
    return "".join(f'<span class="{type_}">{n}</span>' for n in numbers)


def metric_card(title, value, subtitle=""):
    return f"""
    <div class="metric-card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        {"<div style='color:#6B7280;font-size:12px'>"+subtitle+"</div>" if subtitle else ""}
    </div>"""


def plotly_theme():
    return dict(
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0A0E1A",
        font_color="#FFFFFF",
    )


# ═══════════════════════════════════════════════════════
#  PAGE PRINCIPALE
# ═══════════════════════════════════════════════════════

st.markdown("# 🎰 EuroMillions · Analyse Biostatistique & Pronostic")
st.markdown("*Identification de clusters, scoring composite et génération de grilles*")

if uploaded_file is None:
    # ── PAGE D'ACCUEIL ────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card("📊 Analyses disponibles", "10+",
                                "Fréquences, retards, paires, dizaines…"),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("🔬 Clustering", "K-Means",
                                "PCA 2D, profils de clusters"),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("🎯 Pronostic", "3 stratégies",
                                "Chaude, froide, équilibrée"),
                    unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 **Chargez votre fichier Excel EuroMillions** dans la barre latérale pour commencer.")

    with st.expander("📋 Format de fichier attendu"):
        st.markdown("""
        Le fichier Excel doit contenir au minimum **7 colonnes numériques** :
        | Colonne | Description | Plage |
        |---------|-------------|-------|
        | N1, N2, N3, N4, N5 | Boules principales | 1 – 50 |
        | E1, E2 | Étoiles | 1 – 12 |
        | Date *(optionnel)* | Date du tirage | JJ/MM/AAAA |

        **Noms de colonnes acceptés :**
        `N1..N5 / E1..E2`, `Boule 1..5 / Etoile 1..2`,
        `Ball 1..5 / Lucky Star 1..2`, colonnes numériques auto-détectées.
        """)
    st.stop()


# ═══════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════
try:
    df = load_and_clean(uploaded_file)
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

if len(df) < 20:
    st.warning("⚠️ Historique trop court (< 20 tirages). Les analyses ne seront pas fiables.")

# ── CACHE DES CALCULS LOURDS ──────────────────────────
@st.cache_data(show_spinner=False)
def run_analysis(df_bytes, n_clusters, recent_window, w_freq, w_retard, w_trend, w_cluster):
    df_local = pd.read_json(df_bytes, orient="split")
    for c in ["N1","N2","N3","N4","N5","E1","E2"]:
        df_local[c] = df_local[c].astype(int)

    freq_data   = compute_frequencies(df_local)
    retards     = compute_retards(df_local)
    pairs_df    = compute_pairs(df_local)
    triplets_df = compute_triplets(df_local)
    gaps        = compute_gaps(df_local)
    trend_50    = sliding_window_trend(df_local, recent_window)
    sums        = sum_distribution(df_local)
    parity      = parity_analysis(df_local)
    decades     = decade_analysis(df_local)
    consec      = consecutive_analysis(df_local)

    features    = build_draw_features(df_local)
    km_result   = fit_kmeans(features, n_clusters)
    profiles    = cluster_profiles(df_local, km_result["labels"])
    elbow       = optimal_k(features)

    heat        = compute_heat_scores(
        df_local, km_result["labels"], recent_window,
        w_freq, w_retard, w_trend, w_cluster)

    chi_b = chi_square_test(freq_data["freq_boules"],
                            freq_data["expected_boule"])
    chi_e = chi_square_test(freq_data["freq_etoiles"],
                            freq_data["expected_etoile"])

    return dict(
        freq=freq_data, retards=retards, pairs=pairs_df,
        triplets=triplets_df, gaps=gaps, trend=trend_50,
        sums=sums, parity=parity, decades=decades, consec=consec,
        km=km_result, profiles=profiles, elbow=elbow,
        heat=heat, chi_b=chi_b, chi_e=chi_e,
    )

with st.spinner("🔄 Calcul en cours…"):
    results = run_analysis(
        df.to_json(orient="split"),
        n_clusters, recent_window,
        w_freq, w_retard, w_trend, w_cluster,
    )

freq      = results["freq"]
retards   = results["retards"]
heat      = results["heat"]
km        = results["km"]
profiles  = results["profiles"]


# ═══════════════════════════════════════════════════════
#  TABS PRINCIPAUX
# ═══════════════════════════════════════════════════════
tabs = st.tabs([
    "🏠 Vue d'ensemble",
    "📊 Fréquences",
    "⏳ Retards",
    "🔥 Heat Map",
    "🔬 Clustering",
    "🎯 Pronostic",
    "🔗 Paires & Triplets",
    "📈 Tendances avancées",
])


# ──────────────────────────────────────────────────────
#  TAB 0 – VUE D'ENSEMBLE
# ──────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## 📋 Résumé de l'historique")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Tirages analysés", f"{len(df):,}"), unsafe_allow_html=True)
    with c2:
        if "Date" in df.columns:
            d_min = df["Date"].min().strftime("%d/%m/%Y") if pd.notna(df["Date"].min()) else "—"
            st.markdown(metric_card("Premier tirage", d_min), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("Premier tirage", "—"), unsafe_allow_html=True)
    with c3:
        if "Date" in df.columns:
            d_max = df["Date"].max().strftime("%d/%m/%Y") if pd.notna(df["Date"].max()) else "—"
            st.markdown(metric_card("Dernier tirage", d_max), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("Dernier tirage", "—"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Clusters détectés", str(n_clusters)), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔟 Derniers tirages")
    last_n = st.slider("Afficher les N derniers tirages", 5, 50, 10)

    display_df = df.tail(last_n)[["N1","N2","N3","N4","N5","E1","E2"] +
                                 (["Date"] if "Date" in df.columns else [])].copy()
    if "Date" in display_df.columns:
        display_df["Date"] = display_df["Date"].dt.strftime("%d/%m/%Y")
    display_df = display_df.iloc[::-1]

    # Rendu html avec boules colorées
    def render_draw(row):
        b = balls_html([row["N1"],row["N2"],row["N3"],row["N4"],row["N5"]])
        e = balls_html([row["E1"],row["E2"]], "star")
        date_str = f"<small style='color:#AAB0C6'>{row['Date']}</small> &nbsp;" if "Date" in row.index else ""
        return f"{date_str}{b} &nbsp; {e}"

    for _, row in display_df.iterrows():
        st.markdown(render_draw(row), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Test d'uniformité (χ²)")
    col_chi1, col_chi2 = st.columns(2)
    chi_b = results["chi_b"]
    chi_e = results["chi_e"]
    with col_chi1:
        color = "🟢" if chi_b["uniform"] else "🔴"
        st.info(f"**Boules** {color}\n\nχ² = {chi_b['chi2']} | p-valeur = {chi_b['p_value']}\n\n"
                f"{'Distribution uniforme ✅' if chi_b['uniform'] else 'Biais statistique détecté ⚠️'}")
    with col_chi2:
        color = "🟢" if chi_e["uniform"] else "🔴"
        st.info(f"**Étoiles** {color}\n\nχ² = {chi_e['chi2']} | p-valeur = {chi_e['p_value']}\n\n"
                f"{'Distribution uniforme ✅' if chi_e['uniform'] else 'Biais statistique détecté ⚠️'}")


# ──────────────────────────────────────────────────────
#  TAB 1 – FRÉQUENCES
# ──────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📊 Fréquences des tirages")

    mode = st.radio("Afficher", ["Boules (1-50)", "Étoiles (1-12)"], horizontal=True)

    if mode == "Boules (1-50)":
        data = freq["freq_boules"]
        expected = freq["expected_boule"]
        color_label = "Boule"
    else:
        data = freq["freq_etoiles"]
        expected = freq["expected_etoile"]
        color_label = "Étoile"

    # Couleur selon fréquence
    colors = ["#FF6B6B" if v > expected * 1.1
              else "#4FC3F7" if v < expected * 0.9
              else "#FFD700"
              for v in data.values]

    fig = go.Figure(go.Bar(
        x=data.index.tolist(),
        y=data.values,
        marker_color=colors,
        text=data.values,
        textposition="outside",
        hovertemplate=f"<b>{color_label} %{{x}}</b><br>Sorties : %{{y}}<extra></extra>",
    ))
    fig.add_hline(y=expected, line_dash="dash", line_color="#888",
                  annotation_text=f"Espérance ({expected:.1f})")
    fig.update_layout(
        title=f"Fréquence des {color_label.lower()}s",
        xaxis_title=color_label, yaxis_title="Nombre de sorties",
        height=450, bargap=0.1,
        **plotly_theme()
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**🔴 Surreprésenté &nbsp;|&nbsp; 🟡 Normal &nbsp;|&nbsp; 🔵 Sous-représenté**")

    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown("#### 🔝 Top 10 les plus sortis")
        top10 = data.sort_values(ascending=False).head(10)
        for num, cnt in top10.items():
            st.markdown(
                f"{balls_html([num], 'ball' if mode.startswith('B') else 'star')} "
                f"<span style='color:#FFD700'>{cnt} fois</span>",
                unsafe_allow_html=True)
    with col_bot:
        st.markdown("#### ❄️ Top 10 les moins sortis")
        bot10 = data.sort_values().head(10)
        for num, cnt in bot10.items():
            st.markdown(
                f"{balls_html([num], 'ball' if mode.startswith('B') else 'star')} "
                f"<span style='color:#4FC3F7'>{cnt} fois</span>",
                unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TAB 2 – RETARDS
# ──────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## ⏳ Analyse des retards")
    st.markdown("*Le retard = nombre de tirages depuis la dernière apparition. "
                "Un retard élevé signifie que la boule n'est pas sortie depuis longtemps.*")

    mode_r = st.radio("Afficher", ["Boules", "Étoiles"], horizontal=True, key="retard_radio")

    if mode_r == "Boules":
        retard_s = retards["retard_boules"]
        ball_type = "ball"
        expected_gap = 10  # 50/5
    else:
        retard_s = retards["retard_etoiles"]
        ball_type = "star"
        expected_gap = 6   # 12/2

    colors_r = ["#FF4444" if v > expected_gap * 2
                else "#FF9900" if v > expected_gap * 1.3
                else "#4FC3F7" if v < expected_gap * 0.5
                else "#FFD700"
                for v in retard_s.values]

    fig_r = go.Figure(go.Bar(
        x=retard_s.index.tolist(),
        y=retard_s.values,
        marker_color=colors_r,
        hovertemplate="<b>N°%{x}</b><br>Retard : %{y} tirages<extra></extra>",
    ))
    fig_r.add_hline(y=expected_gap, line_dash="dash", line_color="#888",
                    annotation_text=f"Retard moyen ({expected_gap})")
    fig_r.add_hline(y=expected_gap * 2, line_dash="dot", line_color="#FF4444",
                    annotation_text="Seuil critique (×2)")
    fig_r.update_layout(
        title=f"Retards des {mode_r.lower()}",
        xaxis_title="Numéro", yaxis_title="Tirages depuis dernière apparition",
        height=450,
        **plotly_theme()
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    top_retard = retard_s.sort_values(ascending=False).head(10)
    st.markdown("#### 🕰️ Les 10 plus grandes absences")
    for num, r in top_retard.items():
        badge = "🔴" if r > expected_gap * 2 else "🟠" if r > expected_gap * 1.3 else "🟡"
        st.markdown(
            f"{badge} {balls_html([num], ball_type)} "
            f"<span style='color:#AAB0C6'>absent depuis <b style='color:#FF6B6B'>{r}</b> tirages</span>",
            unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TAB 3 – HEAT MAP
# ──────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🔥 Heat Map de chaleur composite")
    st.markdown(f"*Score composite : fréquence ({w_freq:.0%}) + retard ({w_retard:.0%}) "
                f"+ tendance ({w_trend:.0%}) + cluster ({w_cluster:.0%})*")

    # Heatmap boules 10×5
    heat_b = heat["boules"].reindex(range(1, 51), fill_value=0)
    matrix = heat_b.values.reshape(10, 5)

    fig_h = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"Col {i+1}" for i in range(5)],
        y=[f"{i*5+1}-{i*5+5}" for i in range(10)],
        colorscale=[[0,"#0A2550"],[0.4,"#FFD700"],[1,"#FF2222"]],
        text=[[f"{(i*5+j+1)}<br>{matrix[i][j]:.3f}"
               for j in range(5)] for i in range(10)],
        texttemplate="%{text}",
        hovertemplate="Boule %{text}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Score"),
    ))
    fig_h.update_layout(
        title="Score de chaleur – Boules (1-50)",
        height=480, **plotly_theme()
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # Heatmap étoiles
    heat_e = heat["etoiles"].reindex(range(1, 13), fill_value=0)

    fig_he = go.Figure(go.Bar(
        x=list(range(1, 13)),
        y=heat_e.values,
        marker_color=[
            f"rgb({int(255*v/heat_e.max())},{int(120*(1-v/heat_e.max()))},0)"
            for v in heat_e.values
        ],
        text=[f"{v:.3f}" for v in heat_e.values],
        textposition="outside",
        hovertemplate="<b>Étoile %{x}</b><br>Score : %{y:.4f}<extra></extra>",
    ))
    fig_he.update_layout(
        title="Score de chaleur – Étoiles (1-12)",
        xaxis_title="Étoile", yaxis_title="Score composite",
        height=350, **plotly_theme()
    )
    st.plotly_chart(fig_he, use_container_width=True)

    col_hot, col_cold = st.columns(2)
    with col_hot:
        st.markdown("#### 🔥 Top 10 boules les plus chaudes")
        for num, sc in heat["boules"].head(10).items():
            st.markdown(
                f"{balls_html([num])} "
                f"<span class='hot'>score {sc:.4f}</span>",
                unsafe_allow_html=True)
    with col_cold:
        st.markdown("#### ❄️ Top 10 boules les plus froides")
        for num, sc in heat["boules"].tail(10).iloc[::-1].items():
            st.markdown(
                f"{balls_html([num])} "
                f"<span class='cold'>score {sc:.4f}</span>",
                unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TAB 4 – CLUSTERING
# ──────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## 🔬 Analyse par Clustering K-Means")
    st.markdown("*Chaque tirage est représenté par un vecteur de 62 dimensions (50 boules + 12 étoiles). "
                "K-Means regroupe les tirages ayant des patterns similaires.*")

    # Elbow curve
    elbow_df = results["elbow"]
    fig_elbow = go.Figure(go.Scatter(
        x=elbow_df["k"], y=elbow_df["inertia"],
        mode="lines+markers",
        marker=dict(size=8, color="#FFD700"),
        line=dict(color="#FFD700", width=2),
    ))
    fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#FF4444",
                        annotation_text=f"k={n_clusters} sélectionné")
    fig_elbow.update_layout(
        title="Méthode du coude (Elbow) – Choix optimal de k",
        xaxis_title="Nombre de clusters (k)",
        yaxis_title="Inertie intra-cluster",
        height=350, **plotly_theme()
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # PCA 2D scatter
    labels = km["labels"]
    pca_xy = km["pca_coords"]

    # Marquer les 20 derniers tirages
    is_recent = np.zeros(len(df), dtype=bool)
    is_recent[-20:] = True

    fig_pca = go.Figure()
    for cl in range(n_clusters):
        mask = labels == cl
        fig_pca.add_trace(go.Scatter(
            x=pca_xy[mask, 0], y=pca_xy[mask, 1],
            mode="markers",
            name=f"Cluster {cl}",
            marker=dict(size=6, opacity=0.6),
            hovertemplate=f"Cluster {cl}<br>%{{text}}<extra></extra>",
            text=[f"Tirage #{i}" for i in df.index[mask]],
        ))

    # Surligner les récents
    rec_mask = is_recent
    fig_pca.add_trace(go.Scatter(
        x=pca_xy[rec_mask, 0], y=pca_xy[rec_mask, 1],
        mode="markers",
        name="20 derniers",
        marker=dict(size=12, symbol="star", color="#FFD700",
                    line=dict(width=1, color="white")),
    ))
    fig_pca.update_layout(
        title=f"Projection PCA 2D des {len(df)} tirages ({km['explained_variance'].sum()*100:.1f}% variance)",
        height=500, **plotly_theme()
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # Profils de clusters
    st.markdown("### 📋 Profils des clusters")
    for _, row in profiles.iterrows():
        cl_id = int(row["cluster"])
        is_last = (labels[-1] == cl_id)
        badge = " ⭐ **(Cluster actif)**" if is_last else ""
        with st.expander(
            f"Cluster {cl_id} – {row['taille']} tirages ({row['pct']}%){badge}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Boules dominantes :**")
                st.markdown(balls_html(row["top_boules"]), unsafe_allow_html=True)
            with c2:
                st.markdown("**Étoiles dominantes :**")
                st.markdown(balls_html(row["top_etoiles"], "star"),
                            unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TAB 5 – PRONOSTIC
# ──────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## 🎯 Grilles de Pronostic")

    strategy_labels = {
        "balanced": "⚖️ Stratégie équilibrée",
        "hot": "🔥 Stratégie boules chaudes",
        "cold": "❄️ Stratégie boules froides",
    }
    st.markdown(f"### {strategy_labels[strategy]}")
    st.markdown(
        "ℹ️ *Scores basés sur : fréquence historique, retard, tendance récente et cluster dominant. "
        "Le hasard reste la règle fondamentale – ces pronostics sont des outils d'aide statistique.*"
    )

    predictions = generate_predictions(heat, n_grilles, strategy)

    for pred in predictions:
        score_pct = min(100, int(pred["score_moyen"] * 50))
        color_bar = "#FF4444" if score_pct > 70 else "#FFD700" if score_pct > 40 else "#4FC3F7"

        st.markdown(f"""
        <div class="pronostic-card">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                <span class="rank-badge">GRILLE #{pred['grille']}</span>
                <div style="flex:1;background:#1E2438;border-radius:8px;height:8px;overflow:hidden">
                    <div style="width:{score_pct}%;height:100%;background:{color_bar};border-radius:8px"></div>
                </div>
                <span style="color:{color_bar};font-size:13px;font-weight:700">Score {pred['score_moyen']:.4f}</span>
            </div>
            <div style="margin-bottom:8px">
                <span style="color:#AAB0C6;font-size:12px;margin-right:8px">⚽ BOULES</span>
                {balls_html(pred['boules'])}
            </div>
            <div>
                <span style="color:#AAB0C6;font-size:12px;margin-right:8px">⭐ ÉTOILES</span>
                {balls_html(pred['etoiles'], 'star')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Export CSV
    export_rows = []
    for p in predictions:
        export_rows.append({
            "Grille": p["grille"],
            "N1": p["boules"][0], "N2": p["boules"][1], "N3": p["boules"][2],
            "N4": p["boules"][3], "N5": p["boules"][4],
            "E1": p["etoiles"][0], "E2": p["etoiles"][1],
            "Score": p["score_moyen"],
        })
    export_df = pd.DataFrame(export_rows)
    csv_data = export_df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        "📥 Télécharger les grilles (CSV)",
        data=csv_data,
        file_name="euromillions_pronostics.csv",
        mime="text/csv",
    )

    # Légende
    with st.expander("📖 Comment interpréter le score ?"):
        st.markdown("""
        Le score composite est calculé ainsi :

        | Composante | Poids | Description |
        |-----------|-------|-------------|
        | **Fréquence** | configurable | Ratio sorties réelles / sorties espérées |
        | **Retard** | configurable | Ratio retard actuel / intervalle moyen théorique |
        | **Tendance récente** | configurable | Fréquence dans la fenêtre glissante |
        | **Cluster** | configurable | Fréquence dans le cluster du dernier tirage |

        Un score élevé = la boule cumule plusieurs indicateurs positifs.
        **Cela ne garantit pas qu'elle sortira.**
        """)


# ──────────────────────────────────────────────────────
#  TAB 6 – PAIRES & TRIPLETS
# ──────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("## 🔗 Paires et triplets les plus fréquents")

    sub_tab = st.radio("Afficher", ["Paires", "Triplets"], horizontal=True)

    if sub_tab == "Paires":
        pairs_df = results["pairs"].head(30)
        fig_p = go.Figure(go.Bar(
            x=[f"{r['Boule A']}-{r['Boule B']}" for _, r in pairs_df.iterrows()],
            y=pairs_df["Fréquence"],
            marker_color="#FFD700",
            hovertemplate="Paire %{x}<br>Co-sorties : %{y}<extra></extra>",
        ))
        fig_p.update_layout(
            title="Top 30 paires de boules co-sortantes",
            xaxis_title="Paire", yaxis_title="Co-occurrences",
            height=400, **plotly_theme()
        )
        st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("### Top 20 paires")
        for _, r in results["pairs"].head(20).iterrows():
            st.markdown(
                f"{balls_html([r['Boule A'], r['Boule B']])} "
                f"<span style='color:#FFD700'>{r['Fréquence']} co-sorties</span>",
                unsafe_allow_html=True)
    else:
        tri_df = results["triplets"].head(20)
        st.markdown("### Top 20 triplets")
        for _, r in tri_df.iterrows():
            st.markdown(
                f"{balls_html([r['Boule A'], r['Boule B'], r['Boule C']])} "
                f"<span style='color:#FFD700'>{r['Fréquence']} co-sorties</span>",
                unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TAB 7 – TENDANCES AVANCÉES
# ──────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("## 📈 Analyses avancées")

    col_a, col_b = st.columns(2)

    # Distribution des sommes
    with col_a:
        sums = results["sums"]
        fig_sum = go.Figure(go.Histogram(
            x=sums.values,
            nbinsx=40,
            marker_color="#FFD700",
            opacity=0.8,
        ))
        mean_s, std_s = sums.mean(), sums.std()
        fig_sum.add_vline(x=mean_s, line_dash="dash", line_color="#FF6B6B",
                          annotation_text=f"Moy. {mean_s:.1f}")
        fig_sum.update_layout(
            title="Distribution des sommes des 5 boules",
            xaxis_title="Somme", yaxis_title="Fréquence",
            height=340, **plotly_theme()
        )
        st.plotly_chart(fig_sum, use_container_width=True)
        st.info(f"Moyenne : **{mean_s:.1f}** | Écart-type : **{std_s:.1f}** | "
                f"Somme idéale : **127.5** (milieu théorique)")

    # Parité
    with col_b:
        parity = results["parity"]
        fig_par = go.Figure(go.Bar(
            x=[f"{v} pair(s)" for v in parity.index],
            y=parity.values,
            marker_color=["#4FC3F7","#66BB6A","#FFD700","#FF9800","#FF5252","#9C27B0"],
        ))
        fig_par.update_layout(
            title="Répartition pairs/impairs par tirage",
            xaxis_title="Nb de boules paires", yaxis_title="Fréquence",
            height=340, **plotly_theme()
        )
        st.plotly_chart(fig_par, use_container_width=True)

    # Dizaines
    decades = results["decades"]
    fig_dec = go.Figure(go.Bar(
        x=decades.index.tolist(),
        y=decades.values,
        marker_color=["#FF6B6B","#FFD700","#66BB6A","#4FC3F7","#AB47BC"],
        text=decades.values,
        textposition="outside",
    ))
    fig_dec.update_layout(
        title="Répartition par dizaines (1-10, 11-20, …, 41-50)",
        xaxis_title="Dizaine", yaxis_title="Fréquence",
        height=350, **plotly_theme()
    )
    st.plotly_chart(fig_dec, use_container_width=True)

    # Consécutifs
    consec = results["consec"]
    st.markdown("### 🔢 Analyse des suites consécutives")
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        st.markdown(metric_card(
            "Moyenne de paires consécutives",
            f"{consec['mean_consecutifs']:.2f}",
            "par tirage"
        ), unsafe_allow_html=True)
    with col_c2:
        consec_dist = consec["distribution"]
        fig_cons = go.Figure(go.Bar(
            x=[f"{v} paire(s)" for v in consec_dist.index],
            y=consec_dist.values,
            marker_color="#4FC3F7",
        ))
        fig_cons.update_layout(
            title="Distribution des suites consécutives",
            height=280, **plotly_theme()
        )
        st.plotly_chart(fig_cons, use_container_width=True)

    # Tendance glissante
    st.markdown(f"### 📉 Tendance sur les {recent_window} derniers tirages vs historique global")
    trend_rec = results["trend"]
    freq_global = freq["freq_boules"]
    expected_g  = freq["expected_boule"]
    expected_r  = recent_window * 5 / 50

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        name="Historique global",
        x=freq_global.index.tolist(), y=freq_global.values,
        marker_color="#444", opacity=0.6,
    ))
    fig_trend.add_trace(go.Scatter(
        name=f"Tendance {recent_window} tirages",
        x=trend_rec.index.tolist(), y=trend_rec.values,
        mode="lines+markers",
        line=dict(color="#FFD700", width=2),
        marker=dict(size=5),
    ))
    fig_trend.update_layout(
        title="Comparaison fréquence globale vs tendance récente",
        height=420, barmode="overlay", **plotly_theme()
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# ── FOOTER ───────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#444;font-size:12px'>"
    "⚠️ Cette application est à vocation statistique et éducative. "
    "Le jeu de hasard ne peut être prédit avec certitude. Jouez de manière responsable."
    "</div>",
    unsafe_allow_html=True,
)
