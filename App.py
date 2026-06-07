import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

st.set_page_config(page_title="Analyse de tirages", layout="wide")
st.title("🔮 Analyse de l'historique de tirages & Pronostic")
st.markdown("Chargez un fichier Excel contenant l'historique des tirages, sélectionnez les colonnes correspondant aux numéros, et explorez les clusters pour établir un pronostic. **Le hasard reste la règle, ceci est un exercice statistique.**")

uploaded_file = st.file_uploader("📁 Choisir un fichier Excel (.xlsx)", type="xlsx")
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Fichier chargé avec succès !")
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        st.stop()

    # Sélection des colonnes de numéros
    all_cols = df.columns.tolist()
    date_col = st.selectbox("Colonne de date (optionnelle)", ["Aucune"] + all_cols, index=0)
    num_cols = st.multiselect("Colonnes des numéros (sélectionnez toutes les colonnes contenant les numéros)", all_cols)
    if not num_cols:
        st.warning("Veuillez sélectionner au moins une colonne de numéros.")
        st.stop()

    # Vérifier que les colonnes sont numériques
    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"La colonne {col} n'est pas numérique. Veuillez corriger le fichier.")
            st.stop()

    # Paramètres de la loterie
    col1, col2 = st.columns(2)
    with col1:
        min_num = st.number_input("Numéro minimum possible", value=1, step=1)
    with col2:
        max_num = st.number_input("Numéro maximum possible", value=49, step=1)
    n_draws = len(df)
    st.write(f"Nombre de tirages dans l'historique : {n_draws}")

    # Extraire la matrice des tirages
    X = df[num_cols].values.astype(int)
    # Vérifier que les valeurs sont dans l'intervalle
    if np.any(X < min_num) or np.any(X > max_num):
        st.error("Certains numéros sont hors de la plage définie. Veuillez vérifier min/max.")
        st.stop()

    # Analyse fréquentielle simple
    all_numbers = X.flatten()
    freq = pd.Series(all_numbers).value_counts().sort_index()
    freq_full = pd.Series(0, index=range(min_num, max_num+1))
    freq_full.update(freq)
    st.subheader("📊 Fréquence des numéros")
    fig, ax = plt.subplots(figsize=(10,4))
    freq_full.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Nombre d'apparitions")
    st.pyplot(fig)

    hot = freq_full.nlargest(5).index.tolist()
    cold = freq_full.nsmallest(5).index.tolist()
    st.write(f"🔥 Numéros les plus fréquents : {', '.join(map(str, hot))}")
    st.write(f"❄️ Numéros les moins fréquents : {', '.join(map(str, cold))}")

    # Clustering
    st.subheader("🔗 Clustering des tirages (K-Means)")
    n_clusters = st.slider("Nombre de clusters", min_value=2, max_value=10, value=3, step=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters

    # PCA pour visualisation 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:,0]
    df['pca2'] = X_pca[:,1]

    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='viridis', ax=ax2)
    ax2.set_title("Projection des tirages après PCA, colorés par cluster")
    st.pyplot(fig2)

    st.markdown("**Interprétation :** Chaque point représente un tirage. Des tirages proches dans cet espace ont des compositions numériques similaires.")

    # Pronostic
    st.subheader("🎯 Pronostic pour le prochain tirage")
    st.markdown("Deux approches sont proposées. Choisissez celle qui vous inspire le plus (ou combinez-les).")

    # Approche 1 : basée sur le cluster du dernier tirage
    last_cluster = clusters[-1]
    st.write(f"Le dernier tirage appartient au **cluster {last_cluster}**.")
    cluster_mask = (clusters == last_cluster)
    cluster_X = X[cluster_mask]
    # Fréquences dans ce cluster
    cluster_numbers = cluster_X.flatten()
    cluster_freq = pd.Series(cluster_numbers).value_counts().sort_index()
    full_idx = pd.Series(0, index=range(min_num, max_num+1))
    full_idx.update(cluster_freq)
    # Normaliser pour obtenir des probabilités
    proba_cluster = full_idx / full_idx.sum()

    # Approche 2 : basée sur le cluster le plus grand (majoritaire)
    largest_cluster = pd.Series(clusters).value_counts().idxmax()
    st.write(f"Le cluster majoritaire est le **cluster {largest_cluster}** (le plus fréquent dans l'historique).")
    cluster_mask2 = (clusters == largest_cluster)
    cluster_X2 = X[cluster_mask2]
    cluster_numbers2 = cluster_X2.flatten()
    cluster_freq2 = pd.Series(cluster_numbers2).value_counts().sort_index()
    full_idx2 = pd.Series(0, index=range(min_num, max_num+1))
    full_idx2.update(cluster_freq2)
    proba_major = full_idx2 / full_idx2.sum()

    # Méthode de tirage pondéré
    def tirage_pondere(proba_series, n=5):
        nums = proba_series.index.tolist()
        probs = proba_series.values
        probs = probs / probs.sum()  # normalisation
        return np.random.choice(nums, size=n, replace=False, p=probs)

    seed = st.number_input("Graine aléatoire (optionnelle, pour reproductibilité)", value=42, step=1)
    np.random.seed(int(seed))
    n_predictions = st.slider("Nombre de pronostics à afficher par approche", 1, 5, 3)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Pronostics selon le cluster du dernier tirage**")
        preds = []
        for i in range(n_predictions):
            pred = sorted(tirage_pondere(proba_cluster, len(num_cols)))
            preds.append(pred)
            st.write(f"Tirage {i+1} : {', '.join(map(str, pred))}")
    with colB:
        st.markdown("**Pronostics selon le cluster majoritaire**")
        preds2 = []
        for i in range(n_predictions):
            pred = sorted(tirage_pondere(proba_major, len(num_cols)))
            preds2.append(pred)
            st.write(f"Tirage {i+1} : {', '.join(map(str, pred))}")

    st.info("Les numéros sont générés en respectant les fréquences observées dans le cluster choisi, sans remise. Ce ne sont que des suggestions statistiques.")

    # Option : export CSV
    if st.button("📥 Exporter les résultats (fréquences, clusters, pronostics)"):
        result_df = freq_full.reset_index()
        result_df.columns = ["Numéro", "Fréquence"]
        result_df["Probabilité_cluster_dernier"] = proba_cluster.values
        result_df["Probabilité_cluster_majoritaire"] = proba_major.values
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger CSV", csv, "analyse_tirages.csv", "text/csv")
else:
    st.info("Veuillez charger un fichier Excel pour commencer.")
