"""
Module de clustering et prédiction – EuroMillions
Approche biostatistique : identification de clusters temporels,
scoring composite et génération de pronostics probabilistes.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import norm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_draw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme chaque tirage en vecteur de 62 features binaires
    (50 boules + 12 étoiles) pour le clustering.
    """
    rows = []
    for _, row in df.iterrows():
        vec = np.zeros(62, dtype=int)
        for c in ["N1","N2","N3","N4","N5"]:
            vec[int(row[c]) - 1] = 1
        for c in ["E1","E2"]:
            vec[50 + int(row[c]) - 1] = 1
        rows.append(vec)
    cols = [f"B{i}" for i in range(1,51)] + [f"E{i}" for i in range(1,13)]
    return pd.DataFrame(rows, columns=cols, index=df.index)


# ─────────────────────────────────────────────
#  CLUSTERING K-MEANS
# ─────────────────────────────────────────────

def fit_kmeans(features: pd.DataFrame, n_clusters: int = 8):
    """Applique K-Means sur les vecteurs de tirages."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return {
        "model": km,
        "scaler": scaler,
        "labels": labels,
        "pca_coords": X_pca,
        "pca_model": pca,
        "explained_variance": pca.explained_variance_ratio_,
        "inertia": km.inertia_,
    }


def optimal_k(features: pd.DataFrame, k_range=range(3, 16)) -> pd.DataFrame:
    """Inertie (elbow method) pour choisir k optimal."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        results.append({"k": k, "inertia": km.inertia_})
    return pd.DataFrame(results)


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Profil de chaque cluster : boules les plus représentées,
    taille, période temporelle.
    """
    df2 = df.copy()
    df2["cluster"] = labels
    profiles = []
    for cl in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == cl]
        all_nums = pd.concat([sub["N1"],sub["N2"],sub["N3"],sub["N4"],sub["N5"]])
        all_stars = pd.concat([sub["E1"],sub["E2"]])
        top_nums = all_nums.value_counts().head(8).index.tolist()
        top_stars = all_stars.value_counts().head(4).index.tolist()
        profiles.append({
            "cluster": int(cl),
            "taille": len(sub),
            "top_boules": top_nums,
            "top_etoiles": top_stars,
            "pct": round(100 * len(sub) / len(df2), 1),
        })
    return pd.DataFrame(profiles)


# ─────────────────────────────────────────────
#  SCORING COMPOSITE (SCORE DE CHALEUR)
# ─────────────────────────────────────────────

def compute_heat_scores(df: pd.DataFrame,
                        labels: np.ndarray,
                        recent_window: int = 30,
                        weight_freq: float = 0.35,
                        weight_retard: float = 0.30,
                        weight_trend: float = 0.20,
                        weight_cluster: float = 0.15) -> dict:
    """
    Score composite pour chaque boule (1-50) et étoile (1-12).
    Combine :
      - Fréquence historique globale
      - Retard (boules en retard = score élevé)
      - Tendance récente (fenêtre glissante)
      - Appartenance au cluster dominant récent
    """
    n = len(df)
    recent = df.tail(recent_window)

    # Cluster le plus récent
    recent_cluster = labels[-1] if len(labels) > 0 else 0
    recent_label_mask = (labels == recent_cluster)
    cluster_df = df[recent_label_mask]

    scores_b = {}
    for num in range(1, 51):
        # 1. Fréquence globale normalisée
        mask_all = ((df["N1"]==num)|(df["N2"]==num)|(df["N3"]==num)|
                    (df["N4"]==num)|(df["N5"]==num))
        freq = mask_all.sum()
        freq_norm = freq / (n * 5 / 50)  # ratio vs espérance

        # 2. Retard normalisé (retard fort → score élevé)
        idx = df[mask_all].index
        retard = n - 1 - int(idx[-1]) if len(idx) > 0 else n
        expected_gap = 50 / 5  # = 10 tirages en moyenne
        retard_score = min(retard / expected_gap, 3.0)  # plafond à 3x

        # 3. Tendance récente
        mask_rec = ((recent["N1"]==num)|(recent["N2"]==num)|(recent["N3"]==num)|
                    (recent["N4"]==num)|(recent["N5"]==num))
        trend = mask_rec.sum() / (recent_window * 5 / 50)

        # 4. Présence dans cluster récent
        if len(cluster_df) > 0:
            mask_cl = ((cluster_df["N1"]==num)|(cluster_df["N2"]==num)|
                       (cluster_df["N3"]==num)|(cluster_df["N4"]==num)|
                       (cluster_df["N5"]==num))
            cl_score = mask_cl.sum() / (len(cluster_df) * 5 / 50)
        else:
            cl_score = 1.0

        scores_b[num] = (weight_freq * freq_norm +
                         weight_retard * retard_score +
                         weight_trend * trend +
                         weight_cluster * cl_score)

    scores_e = {}
    for num in range(1, 13):
        mask_all = (df["E1"]==num)|(df["E2"]==num)
        freq = mask_all.sum()
        freq_norm = freq / (n * 2 / 12)

        idx = df[mask_all].index
        retard = n - 1 - int(idx[-1]) if len(idx) > 0 else n
        expected_gap = 12 / 2
        retard_score = min(retard / expected_gap, 3.0)

        mask_rec = (recent["E1"]==num)|(recent["E2"]==num)
        trend = mask_rec.sum() / (recent_window * 2 / 12)

        if len(cluster_df) > 0:
            mask_cl = (cluster_df["E1"]==num)|(cluster_df["E2"]==num)
            cl_score = mask_cl.sum() / (len(cluster_df) * 2 / 12)
        else:
            cl_score = 1.0

        scores_e[num] = (weight_freq * freq_norm +
                         weight_retard * retard_score +
                         weight_trend * trend +
                         weight_cluster * cl_score)

    return {
        "boules": pd.Series(scores_b).sort_values(ascending=False),
        "etoiles": pd.Series(scores_e).sort_values(ascending=False),
    }


# ─────────────────────────────────────────────
#  GÉNÉRATION DES PRONOSTICS
# ─────────────────────────────────────────────

def generate_predictions(heat_scores: dict,
                         n_grilles: int = 5,
                         strategy: str = "balanced") -> list:
    """
    Génère n_grilles de pronostics selon la stratégie choisie :
      - "hot"      : top boules les plus chaudes
      - "cold"     : boules les plus froides (en retard maximal)
      - "balanced" : mix chaud/froid (recommandé)
      - "cluster"  : basé uniquement sur le cluster dominant
    """
    boules_scores = heat_scores["boules"]
    etoiles_scores = heat_scores["etoiles"]

    grilles = []

    for g in range(n_grilles):
        if strategy == "hot":
            # Top-N avec légère variation aléatoire pondérée
            boules = _weighted_sample(boules_scores, 5, top_bias=0.8, seed=g)
            etoiles = _weighted_sample(etoiles_scores, 2, top_bias=0.8, seed=g+100)

        elif strategy == "cold":
            inv_b = 1 / (boules_scores + 0.01)
            inv_e = 1 / (etoiles_scores + 0.01)
            boules = _weighted_sample(inv_b, 5, top_bias=0.7, seed=g+200)
            etoiles = _weighted_sample(inv_e, 2, top_bias=0.7, seed=g+300)

        elif strategy == "balanced":
            # 3 boules chaudes + 2 boules froides
            hot_b = boules_scores.head(20)
            cold_b = boules_scores.tail(20)
            hot_e = etoiles_scores.head(6)
            cold_e = etoiles_scores.tail(6)

            rng = np.random.default_rng(seed=42 + g)
            hot_pick_b = _weighted_sample(hot_b, 3, top_bias=0.65, seed=g*7)
            remaining_b = boules_scores.drop(hot_pick_b)
            cold_pick_b = _weighted_sample(
                remaining_b.tail(20), 2, top_bias=0.5, seed=g*13)
            boules = sorted(hot_pick_b + cold_pick_b)

            hot_pick_e = _weighted_sample(hot_e, 1, top_bias=0.7, seed=g*3)
            remaining_e = etoiles_scores.drop(hot_pick_e)
            cold_pick_e = _weighted_sample(remaining_e, 1, top_bias=0.5, seed=g*17)
            etoiles = sorted(hot_pick_e + cold_pick_e)

        else:  # cluster-based fallback = hot
            boules = _weighted_sample(boules_scores, 5, top_bias=0.75, seed=g+500)
            etoiles = _weighted_sample(etoiles_scores, 2, top_bias=0.75, seed=g+600)

        grilles.append({
            "grille": g + 1,
            "boules": sorted(boules),
            "etoiles": sorted(etoiles),
            "score_moyen": round(
                np.mean([boules_scores.get(b, 0) for b in boules] +
                        [etoiles_scores.get(e, 0) for e in etoiles]), 4)
        })

    # Tri par score décroissant
    grilles.sort(key=lambda x: x["score_moyen"], reverse=True)
    return grilles


def _weighted_sample(scores: pd.Series, k: int,
                     top_bias: float = 0.7, seed: int = 0) -> list:
    """Tirage pondéré par les scores, avec biais vers le top."""
    rng = np.random.default_rng(seed=seed)
    s = scores.copy()
    s = s - s.min()          # min = 0
    if s.sum() == 0:
        s[:] = 1
    probs = (s ** top_bias) / (s ** top_bias).sum()
    chosen = rng.choice(s.index.tolist(), size=k, replace=False, p=probs.values)
    return sorted(chosen.tolist())


# ─────────────────────────────────────────────
#  ANALYSE DE SÉQUENCES (MARKOV SIMPLIFIÉ)
# ─────────────────────────────────────────────

def markov_transition(df: pd.DataFrame, num_range: range = range(1, 51)) -> pd.DataFrame:
    """
    Matrice de transition : P(num apparaît | num est apparu au tirage précédent).
    Donne une idée des 'suites' de boules.
    """
    n = len(df)
    trans = np.zeros((len(num_range), len(num_range)))

    for i in range(1, n):
        prev = set([df.iloc[i-1]["N1"], df.iloc[i-1]["N2"],
                    df.iloc[i-1]["N3"], df.iloc[i-1]["N4"],
                    df.iloc[i-1]["N5"]])
        curr = set([df.iloc[i]["N1"], df.iloc[i]["N2"],
                    df.iloc[i]["N3"], df.iloc[i]["N4"],
                    df.iloc[i]["N5"]])
        for p in prev:
            for c in curr:
                trans[p-1][c-1] += 1

    # Normalisation par ligne
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = trans / row_sums

    return pd.DataFrame(trans,
                        index=list(num_range),
                        columns=list(num_range))


def consecutive_analysis(df: pd.DataFrame) -> dict:
    """Analyse des séquences consécutives dans les tirages."""
    consec_counts = []
    for _, row in df.iterrows():
        nums = sorted([row["N1"],row["N2"],row["N3"],row["N4"],row["N5"]])
        consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
        consec_counts.append(consec)
    s = pd.Series(consec_counts)
    return {
        "mean_consecutifs": round(s.mean(), 3),
        "distribution": s.value_counts().sort_index(),
    }
