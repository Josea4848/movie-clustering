import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


def getScenes(N, csv_path="frames_dataset.csv"):
    """ 
    Retorna um DataFrame com os N frames mais significativos do maior cluster de cada filme. 

    Args: 
        N (int): Número de frames mais significativos a serem extraídos por filme. 
        csv_path (str): Caminho para o CSV com os dados dos frames. 
    
    Returns:
        pd.DataFrame: DataFrame com todas as colunas originais + cluster.

    """

    df = pd.read_csv(csv_path)
    df = df[df["path"].str.contains("frame_1.jpg", regex=False)]

    df = df[df["movie"].isin(["MoonriseKingdom2012", "AsteroidCity2023", "ForAFewDollarsMore1965", "TheGoodTheBadAndTheUgly1966"])]

    features = ["mean_r", "mean_g", "mean_b", "mean_saturation",
                "mean_luminosity", "contrast", "temp_color", "Laplacian_variance"]

    for col in features:
        df = df[pd.to_numeric(df[col], errors='coerce').notna()]
        df[col] = df[col].astype(float)

    result_rows = []

    for movie in df['movie'].unique():
        df_movie = df[df['movie'] == movie].copy()
        X = df_movie[features].values

        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN
        dbscan = DBSCAN(eps=2.0, min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)
        df_movie['cluster'] = clusters

        # Maior cluster
        clusters_unique = [c for c in set(clusters) if c != -1]
        if not clusters_unique:
            continue
        cluster_sizes = {c: sum(df_movie['cluster'] == c) for c in clusters_unique}
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)

        cluster_data = X_scaled[df_movie['cluster'] == largest_cluster]
        cluster_indices = df_movie[df_movie['cluster'] == largest_cluster].index.tolist()

        # Top N frames mais próximos do centróide
        centroid = cluster_data.mean(axis=0)
        distances = np.linalg.norm(cluster_data - centroid, axis=1)
        top_indices = distances.argsort()[:N]

        for idx in top_indices:
            row_index = cluster_indices[idx]
            result_rows.append(df_movie.loc[row_index])

    return pd.DataFrame(result_rows).reset_index(drop=True)



# Extrair frames
result = getScenes(500)


# Features e normalização
features = ["mean_r", "mean_g", "mean_b", "mean_saturation",
            "mean_luminosity", "contrast", "temp_color", "Laplacian_variance"]
X = result[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
result["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
result["pca1"] = X_pca[:, 0]
result["pca2"] = X_pca[:, 1]


# Comparação com classes reais
conf_matrix = pd.crosstab(result['movie'], result['kmeans_cluster'])
print("Matriz de Confusão:")
print(conf_matrix)

ari = adjusted_rand_score(result['movie'], result['kmeans_cluster'])
print(f"\nAdjusted Rand Index (ARI): {ari:.3f}")


# Visualização PCA
plt.figure(figsize=(12, 5))

# Por filme
plt.subplot(1, 2, 1)
for movie in result['movie'].unique():
    subset = result[result['movie'] == movie]
    plt.scatter(subset['pca1'], subset['pca2'], label=movie, alpha=0.7)
plt.title("PCA 2D - Por Filme")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Por cluster
plt.subplot(1, 2, 2)
for cluster in result['kmeans_cluster'].unique():
    subset = result[result['kmeans_cluster'] == cluster]
    plt.scatter(subset['pca1'], subset['pca2'], label=f"Cluster {cluster}", alpha=0.7)
plt.title("PCA 2D - Por Cluster")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# Salvar CSV final
result.to_csv("best_scenes_kmeans_pca_comparison.csv", index=False)
