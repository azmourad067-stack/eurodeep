import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import io
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(page_title="EuroMillions Predictor", layout="wide")
st.title("🎱 EuroMillions Statistical Predictor")
st.markdown("**Based on historical draws – identifies clusters and trends to suggest next likely numbers**")

# ------------------------------
# Helper functions
# ------------------------------
@st.cache_data
def load_data(uploaded_file):
    """Load CSV (semicolon) or Excel file, extract draws."""
    if uploaded_file.name.endswith('.csv'):
        # Try to read with semicolon delimiter, skip bad lines
        df = pd.read_csv(uploaded_file, delimiter=';', on_bad_lines='skip', dtype=str)
    else:
        df = pd.read_excel(uploaded_file, dtype=str)

    # Keep only rows where boule_1 is a number (skip empty/header rows)
    df = df[df['boule_1'].str.isnumeric()].copy()
    # Convert relevant columns to int
    number_cols = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
    star_cols = ['etoile_1', 'etoile_2']
    for col in number_cols + star_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=number_cols + star_cols).reset_index(drop=True)
    return df

def compute_frequencies(df):
    """Return frequency Series for numbers (1-50) and stars (1-12)."""
    all_numbers = df[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.flatten()
    all_stars = df[['etoile_1','etoile_2']].values.flatten()
    num_counts = Counter(all_numbers)
    star_counts = Counter(all_stars)
    # Fill missing numbers (1..50, 1..12) with 0
    num_freq = pd.Series({i: num_counts.get(i,0) for i in range(1,51)})
    star_freq = pd.Series({i: star_counts.get(i,0) for i in range(1,13)})
    return num_freq, star_freq

def co_occurrence_matrix(df, col_list, max_val):
    """Build co-occurrence matrix for numbers (50x50) or stars (12x12)."""
    mat = np.zeros((max_val, max_val), dtype=int)
    for _, row in df.iterrows():
        combo = row[col_list].values.astype(int)
        for i in range(len(combo)):
            for j in range(i+1, len(combo)):
                a, b = combo[i]-1, combo[j]-1
                mat[a,b] += 1
                mat[b,a] += 1
    return mat

def predict_next(num_freq, star_freq, df, n_draws=20):
    """
    Weighted random prediction using frequency + recency boost.
    Numbers that have not appeared for many draws get a small extra weight.
    """
    last_draws = df.tail(n_draws)
    last_numbers = last_draws[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.flatten()
    last_stars = last_draws[['etoile_1','etoile_2']].values.flatten()

    # Base weight = frequency (normalized)
    num_weight = num_freq.values.copy().astype(float)
    star_weight = star_freq.values.copy().astype(float)

    # Recency boost: if a number didn't appear in last n_draws, increase its weight by 20%
    for i in range(1, 51):
        if i not in last_numbers:
            num_weight[i-1] *= 1.2
    for i in range(1, 13):
        if i not in last_stars:
            star_weight[i-1] *= 1.2

    # Add small epsilon to avoid zero probabilities
    num_weight = np.maximum(num_weight, 0.01)
    star_weight = np.maximum(star_weight, 0.01)

    # Normalize to probabilities
    num_prob = num_weight / num_weight.sum()
    star_prob = star_weight / star_weight.sum()

    # Draw 5 distinct numbers and 2 distinct stars
    predicted_numbers = set()
    while len(predicted_numbers) < 5:
        pick = np.random.choice(range(1,51), p=num_prob)
        predicted_numbers.add(pick)
    predicted_stars = set()
    while len(predicted_stars) < 2:
        pick = np.random.choice(range(1,13), p=star_prob)
        predicted_stars.add(pick)

    return sorted(predicted_numbers), sorted(predicted_stars)

# ------------------------------
# UI: File uploader
# ------------------------------
uploaded_file = st.file_uploader("Upload your EuroMillions history (CSV or Excel)", type=['csv','xlsx'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success(f"Loaded {len(df)} draws")

    # Sidebar options
    st.sidebar.header("Analysis Options")
    n_last = st.sidebar.slider("Number of recent draws for recency boost", 5, 50, 20)

    # Compute frequencies
    num_freq, star_freq = compute_frequencies(df)

    # Display basic stats
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Most Frequent Main Numbers")
        top_nums = num_freq.sort_values(ascending=False).head(10)
        st.bar_chart(top_nums)
    with col2:
        st.subheader("⭐ Most Frequent Stars")
        top_stars = star_freq.sort_values(ascending=False).head(5)
        st.bar_chart(top_stars)

    # Co-occurrence heatmap (optional, can be slow, so we limit)
    if st.checkbox("Show co‑occurrence heatmap (main numbers)", value=False):
        st.subheader("Numbers Co‑occurrence Matrix")
        max_num = 50
        co_num = co_occurrence_matrix(df, ['boule_1','boule_2','boule_3','boule_4','boule_5'], max_num)
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(co_num, cmap='Blues', ax=ax, xticklabels=False, yticklabels=False)
        ax.set_title("Pairwise co‑occurrence of main numbers")
        st.pyplot(fig)

    # Prediction button
    if st.button("🔮 Generate Next Draw Prediction"):
        pred_numbers, pred_stars = predict_next(num_freq, star_freq, df, n_last)
        st.subheader("🎯 Predicted Combination for the Next Draw")
        cols = st.columns(5)
        for i, num in enumerate(pred_numbers):
            cols[i].metric(label=f"Number {i+1}", value=num)
        col_stars = st.columns(2)
        for i, star in enumerate(pred_stars):
            col_stars[i].metric(label=f"Star {i+1}", value=star)

        st.caption("Prediction is based on historical frequencies + a recency boost. "
                   "Lotteries are random; this is a statistical exercise, not a guarantee.")

    # Optional: show cold numbers
    with st.expander("❄️ Cold numbers (rarely drawn)"):
        cold_nums = num_freq[num_freq < num_freq.quantile(0.2)].sort_values()
        cold_stars = star_freq[star_freq < star_freq.quantile(0.2)].sort_values()
        st.write("Main numbers that appear least often:", list(cold_nums.index))
        st.write("Stars that appear least often:", list(cold_stars.index))

    # Show raw data sample
    if st.checkbox("Show raw data sample"):
        st.dataframe(df.head(20))

else:
    st.info("👈 Please upload a CSV or Excel file (example format provided) to begin.")
    st.markdown("""
    **Expected file format** (semicolon‑separated CSV or Excel):
    - Columns: `boule_1`, `boule_2`, `boule_3`, `boule_4`, `boule_5`, `etoile_1`, `etoile_2`
    - One row per draw.
    - The example file you attached works perfectly.
    """)
