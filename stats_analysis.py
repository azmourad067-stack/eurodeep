"""
Module d'analyse statistique des tirages EuroMillions
Biostatistique & Analyse du hasard
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  CHARGEMENT & NETTOYAGE DES DONNÉES
# ─────────────────────────────────────────────

def load_and_clean(file) -> pd.DataFrame:
    """
    Charge un fichier Excel EuroMillions et normalise les colonnes.
    Formats supportés :
      - Colonnes nommées N1..N5, E1/E2 (ou Boule1..5, Etoile1/2)
      - Première ligne = en-tête ou données brutes
    """
    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception:
        df = pd.read_excel(file, engine="xlrd")

    df.columns = [str(c).strip() for c in df.columns]

    # --- Détection automatique des colonnes ---
    col_map = _detect_columns(df)
    if col_map is None:
        raise ValueError(
            "Impossible de détecter les colonnes de tirage.\n"
            "Le fichier doit contenir 5 colonnes de boules (1-50) "
            "et 2 colonnes d'étoiles (1-12)."
        )

    # Renommage standardisé
    df = df.rename(columns=col_map)

    # Colonnes obligatoires
    required = ["N1", "N2", "N3", "N4", "N5", "E1", "E2"]

    # Colonne date optionnelle
    date_col = _find_date_col(df)
    if date_col:
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        keep = required + ["Date"]
    else:
        keep = required

    df = df[keep].copy()

    # Conversion numérique + suppression des lignes invalides
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required)
    for c in required:
        df[c] = df[c].astype(int)

    # Validation des plages
    boules_ok = df[["N1","N2","N3","N4","N5"]].apply(
        lambda col: col.between(1, 50)).all(axis=1)
    etoiles_ok = df[["E1","E2"]].apply(
        lambda col: col.between(1, 12)).all(axis=1)
    df = df[boules_ok & etoiles_ok].reset_index(drop=True)

    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def _detect_columns(df: pd.DataFrame):
    """Détecte et mappe les colonnes vers N1-N5, E1-E2."""
    cols = df.columns.tolist()

    # Patterns connus
    patterns = [
        # EuroMillions standard français
        {"N1":"Boule 1","N2":"Boule 2","N3":"Boule 3","N4":"Boule 4","N5":"Boule 5","E1":"Etoile 1","E2":"Etoile 2"},
        {"N1":"boule_1","N2":"boule_2","N3":"boule_3","N4":"boule_4","N5":"boule_5","E1":"etoile_1","E2":"etoile_2"},
        {"N1":"N1","N2":"N2","N3":"N3","N4":"N4","N5":"N5","E1":"E1","E2":"E2"},
        {"N1":"Ball 1","N2":"Ball 2","N3":"Ball 3","N4":"Ball 4","N5":"Ball 5","E1":"Lucky Star 1","E2":"Lucky Star 2"},
        {"N1":"Num1","N2":"Num2","N3":"Num3","N4":"Num4","N5":"Num5","E1":"Star1","E2":"Star2"},
    ]
    cols_lower = [c.lower().replace(" ","_") for c in cols]

    for pat in patterns:
        vals_lower = [v.lower().replace(" ","_") for v in pat.values()]
        if all(v in cols_lower for v in vals_lower):
            idx = {v: cols_lower.index(v) for v in vals_lower}
            return {cols[idx[v.lower().replace(" ","_")]]: k for k, v in pat.items()}

    # Détection heuristique : cherche 7 colonnes numériques consécutives
    num_cols = []
    for c in cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) > len(df) * 0.7:
                num_cols.append(c)
        except Exception:
            pass

    if len(num_cols) >= 7:
        boule_cols, star_cols = [], []
        for c in num_cols:
            med = pd.to_numeric(df[c], errors="coerce").median()
            if 1 <= med <= 50:
                boule_cols.append(c)
            if 1 <= med <= 12:
                star_cols.append(c)
        # Priorité boules longues
        boule_cols = sorted(set(boule_cols) - set(star_cols[:2]),
                            key=lambda c: num_cols.index(c))
        if len(boule_cols) >= 5 and len(star_cols) >= 2:
            mapping = {}
            for i, c in enumerate(boule_cols[:5]):
                mapping[c] = f"N{i+1}"
            for i, c in enumerate(star_cols[:2]):
                mapping[c] = f"E{i+1}"
            return mapping

    return None


def _find_date_col(df: pd.DataFrame):
    for c in df.columns:
        if any(kw in c.lower() for kw in ["date","jour","draw","tirage"]):
            return c
    return None


# ─────────────────────────────────────────────
#  STATISTIQUES DE BASE
# ─────────────────────────────────────────────

def compute_frequencies(df: pd.DataFrame) -> dict:
    """Fréquences absolues et relatives des boules et étoiles."""
    boules = pd.concat([df["N1"],df["N2"],df["N3"],df["N4"],df["N5"]])
    etoiles = pd.concat([df["E1"],df["E2"]])

    freq_b = boules.value_counts().reindex(range(1,51), fill_value=0)
    freq_e = etoiles.value_counts().reindex(range(1,13), fill_value=0)

    n_draws = len(df)
    return {
        "freq_boules": freq_b,
        "freq_etoiles": freq_e,
        "rel_boules": freq_b / (n_draws * 5),
        "rel_etoiles": freq_e / (n_draws * 2),
        "n_draws": n_draws,
        "expected_boule": n_draws * 5 / 50,
        "expected_etoile": n_draws * 2 / 12,
    }


def compute_retards(df: pd.DataFrame) -> dict:
    """
    Retard = nombre de tirages depuis la dernière apparition.
    Plus le retard est grand, plus la boule est 'en retard'.
    """
    n = len(df)
    retard_b, retard_e = {}, {}

    for num in range(1, 51):
        mask = ((df["N1"]==num)|(df["N2"]==num)|(df["N3"]==num)|
                (df["N4"]==num)|(df["N5"]==num))
        idx = df[mask].index
        retard_b[num] = n - 1 - int(idx[-1]) if len(idx) > 0 else n

    for num in range(1, 13):
        mask = (df["E1"]==num)|(df["E2"]==num)
        idx = df[mask].index
        retard_e[num] = n - 1 - int(idx[-1]) if len(idx) > 0 else n

    return {
        "retard_boules": pd.Series(retard_b),
        "retard_etoiles": pd.Series(retard_e),
    }


def compute_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Paires de boules les plus fréquentes."""
    pair_count = Counter()
    for _, row in df.iterrows():
        nums = sorted([row["N1"],row["N2"],row["N3"],row["N4"],row["N5"]])
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pair_count[(nums[i], nums[j])] += 1
    pairs_df = pd.DataFrame(
        [(a, b, c) for (a,b),c in pair_count.items()],
        columns=["Boule A","Boule B","Fréquence"]
    ).sort_values("Fréquence", ascending=False).reset_index(drop=True)
    return pairs_df


def compute_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Triplets les plus fréquents."""
    tri_count = Counter()
    for _, row in df.iterrows():
        nums = sorted([row["N1"],row["N2"],row["N3"],row["N4"],row["N5"]])
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                for k in range(j+1, len(nums)):
                    tri_count[(nums[i], nums[j], nums[k])] += 1
    tri_df = pd.DataFrame(
        [(a, b, c, n) for (a,b,c),n in tri_count.items()],
        columns=["Boule A","Boule B","Boule C","Fréquence"]
    ).sort_values("Fréquence", ascending=False).reset_index(drop=True)
    return tri_df


def compute_gaps(df: pd.DataFrame) -> dict:
    """Intervalles entre apparitions successives de chaque boule."""
    gaps_b = {}
    for num in range(1, 51):
        mask = ((df["N1"]==num)|(df["N2"]==num)|(df["N3"]==num)|
                (df["N4"]==num)|(df["N5"]==num))
        idx = df[mask].index.tolist()
        if len(idx) > 1:
            g = [idx[i+1]-idx[i] for i in range(len(idx)-1)]
            gaps_b[num] = {"mean": np.mean(g), "std": np.std(g),
                           "min": min(g), "max": max(g)}
        else:
            gaps_b[num] = {"mean": None, "std": None, "min": None, "max": None}

    gaps_e = {}
    for num in range(1, 13):
        mask = (df["E1"]==num)|(df["E2"]==num)
        idx = df[mask].index.tolist()
        if len(idx) > 1:
            g = [idx[i+1]-idx[i] for i in range(len(idx)-1)]
            gaps_e[num] = {"mean": np.mean(g), "std": np.std(g),
                           "min": min(g), "max": max(g)}
        else:
            gaps_e[num] = {"mean": None, "std": None, "min": None, "max": None}

    return {"gaps_boules": gaps_b, "gaps_etoiles": gaps_e}


def chi_square_test(freq: pd.Series, expected: float) -> dict:
    """Test du χ² d'adéquation à une distribution uniforme."""
    observed = freq.values
    exp_arr = np.full(len(observed), expected)
    chi2, p = stats.chisquare(observed, f_exp=exp_arr)
    return {"chi2": round(chi2, 4), "p_value": round(p, 6),
            "uniform": p > 0.05}


def sliding_window_trend(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Tendance glissante sur les `window` derniers tirages.
    Retourne la fréquence de chaque boule dans cette fenêtre.
    """
    recent = df.tail(window)
    boules = pd.concat([recent["N1"],recent["N2"],recent["N3"],
                        recent["N4"],recent["N5"]])
    freq = boules.value_counts().reindex(range(1,51), fill_value=0)
    return freq


def sum_distribution(df: pd.DataFrame) -> pd.Series:
    """Distribution de la somme des 5 boules par tirage."""
    return (df["N1"]+df["N2"]+df["N3"]+df["N4"]+df["N5"])


def parity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse pair/impair des boules."""
    rows = []
    for _, row in df.iterrows():
        nums = [row["N1"],row["N2"],row["N3"],row["N4"],row["N5"]]
        evens = sum(1 for n in nums if n % 2 == 0)
        rows.append(evens)
    s = pd.Series(rows)
    return s.value_counts().sort_index()


def decade_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Répartition par dizaines (1-10, 11-20, 21-30, 31-40, 41-50)."""
    all_nums = pd.concat([df["N1"],df["N2"],df["N3"],df["N4"],df["N5"]])
    bins = [0,10,20,30,40,50]
    labels = ["1-10","11-20","21-30","31-40","41-50"]
    return pd.cut(all_nums, bins=bins, labels=labels).value_counts().sort_index()
