import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from Utils import show_path_structure, visualize_selected_frames

def getScenes(N : int, filter_movies : list, csv_path="frames_dataset.csv",):
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
    
    # Verifica se possui filtro para filmes
    if len(filter_movies):
        df = df[df["movie"].isin(filter_movies)]
       
    features = ["mean_hue", "mean_saturation", "mean_brightness",
            "contrast", "temperature", "sharpness", "edge_density", "entropy"]

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

        # Remove ruído
        clusters_unique = [c for c in set(clusters) if c != -1]
        if not clusters_unique:
            continue

        # Maior cluster
        cluster_sizes = {c: sum(df_movie['cluster'] == c) for c in clusters_unique}
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)

        # Filtra os frames que pertencem ao maior cluster
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

def main():
    print("Extraindo frames mais significativos...")
    result = getScenes(300, ["Batman1989", "BatmanReturns1992", "MadMaxFuryRoad2015"])
    print(f"Extração concluída")
    print(f"Total de frames selecionados: {len(result)}")
    print(f"Distribuição por filme: {result['movie'].value_counts()}")

    # Mostrar estrutura dos caminhos
    show_path_structure(result)

    features = ["mean_hue", "mean_saturation", "mean_brightness",
                "contrast", "temperature", "sharpness", "edge_density", "entropy"]
    
    # Normalizando valores
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

    # Matriz de confusão
    conf_matrix = pd.crosstab(result['movie'], result['kmeans_cluster'])
    print("Matriz de Confusão (Filme vs Cluster K-Means):")
    print(conf_matrix)

    ari = adjusted_rand_score(result['director'], result['kmeans_cluster'])
    print(f"\nAdjusted Rand Index (ARI): {ari:.3f}")

    # Visualização PCA
    plt.figure(figsize=(15, 6))

    # Por filme
    plt.subplot(1, 2, 1)
    for movie in result['movie'].unique():
        subset = result[result['movie'] == movie]
        plt.scatter(subset['pca1'], subset['pca2'], label=movie, alpha=0.7, s=50)
    plt.title("PCA 2D - Agrupamento por Filme", fontsize=14)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)

    # Por cluster
    plt.subplot(1, 2, 2)
    for cluster in result['kmeans_cluster'].unique():
        subset = result[result['kmeans_cluster'] == cluster]
        plt.scatter(subset['pca1'], subset['pca2'], label=f"Cluster {cluster}", alpha=0.7, s=50)
    plt.title("PCA 2D - Agrupamento por Cluster K-Means", fontsize=14)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Mostrar estatísticas finais
    print(f"Total de frames processados: {len(result)}")
    print(f"Número de clusters K-Means: {result['kmeans_cluster'].nunique()}")
    print(f"Filmes analisados: {len(result['movie'].unique())}")
    
    for movie in result['movie'].unique():
        count = len(result[result['movie'] == movie])
        print(f"  - {movie}: {count} frames")

    # Visualizar frames selecionados
    visualize_selected_frames(result, num_frames_per_movie=10, base_path="")

if __name__ == "__main__":
    main()