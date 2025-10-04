import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

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
    #df = df[df["movie"].isin(["Batman1989", "BatmanReturns1992",  "AHiddenLife2019","KnightOfCups2015", "EliteSquad2007", "EliteSquad2TheEnemyWithin2010", "OnceUponaTimeintheWest1968", "TheGoodTheBadAndTheUgly1966"])]

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

def visualize_selected_frames(result_df, num_frames_per_movie=5, base_path=""):
    """
    Visualiza os frames selecionados para cada filme mostrando o caminho completo.
    
    Args:
        result_df (pd.DataFrame): DataFrame retornado por getScenes
        num_frames_per_movie (int): Número de frames para mostrar por filme
        base_path (str): Caminho base para as imagens, se necessário
    """
    movies = result_df['movie'].unique()
    
    for movie in movies:
        print(f"\n" + "="*80)
        print(f"FILME: {movie}")
        print("="*80)
        
        movie_frames = result_df[result_df['movie'] == movie]
        
        # Pegar os primeiros N frames para visualização
        sample_frames = movie_frames.head(num_frames_per_movie)
        
        print(f"Total de frames selecionados para este filme: {len(movie_frames)}")
        print(f"Mostrando os primeiros {len(sample_frames)} frames:\n")
        
        fig, axes = plt.subplots(1, len(sample_frames), figsize=(18, 4))
        if len(sample_frames) == 1:
            axes = [axes]
        
        for idx, (_, row) in enumerate(sample_frames.iterrows()):
            img_path = row['path']
            
            # Mostrar informações completas do path
            print(f"FRAME {idx+1}:")
            print(f"  Caminho completo: {img_path}")
            print(f"  Cluster DBSCAN: {row['cluster']}")
            print(f"  Características - Hue: {row['mean_hue']:.2f}, Sat: {row['mean_saturation']:.2f}, "
                  f"Bright: {row['mean_brightness']:.2f}, Contrast: {row['contrast']:.2f}")
            print("-" * 50)
            
            # Ajustar caminho se necessário
            full_path = img_path
            if base_path and not os.path.exists(img_path):
                full_path = os.path.join(base_path, img_path)
            
            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[idx].imshow(img)
                axes[idx].set_title(f"{movie}\nFrame {idx+1}\nCluster: {row['cluster']}", fontsize=10)
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, f"Imagem não encontrada:\n{os.path.basename(img_path)}", 
                              ha='center', va='center', transform=axes[idx].transAxes, fontsize=8)
                axes[idx].axis('off')
                print(f"  AVISO: Imagem não encontrada em: {full_path}")
        
        plt.tight_layout()
        plt.show()

def show_path_structure(result_df):
    """
    Mostra a estrutura dos caminhos para entender melhor a organização dos arquivos.
    """
    print("\n" + "="*80)
    print("ESTRUTURA DOS CAMINHOS DOS FRAMES SELECIONADOS")
    print("="*80)
    
    for movie in result_df['movie'].unique():
        movie_frames = result_df[result_df['movie'] == movie]
        print(f"\n🎬 {movie} ({len(movie_frames)} frames):")
        
        # Mostrar alguns caminhos de exemplo
        sample_paths = movie_frames['path'].head(3).tolist()
        for i, path in enumerate(sample_paths, 1):
            print(f"  Exemplo {i}: {path}")
        
        # Analisar estrutura de diretórios
        dirs = [os.path.dirname(path) for path in movie_frames['path'].head(5)]
        unique_dirs = list(set(dirs))
        if unique_dirs:
            print(f"  Diretórios encontrados: {unique_dirs[0]}")
            if len(unique_dirs) > 1:
                print(f"  (+ mais {len(unique_dirs)-1} diretórios diferentes)")

# Extrair frames
print("Extraindo frames mais significativos...")
result = getScenes(500)
print(f"\nExtração concluída!")
print(f"Total de frames selecionados: {len(result)}")
print(f"Distribuição por filme:")
print(result['movie'].value_counts())

# Mostrar estrutura dos caminhos
show_path_structure(result)

# Visualizar alguns frames de cada filme
print("\n" + "="*80)
print("VISUALIZANDO FRAMES SELECIONADOS")
print("="*80)

# Ajuste o base_path conforme necessário para o seu ambiente
base_path = ""  # Coloque o caminho base se suas imagens estiverem em subdiretórios
#visualize_selected_frames(result, num_frames_per_movie=3, base_path=base_path)

# Análise de clusters (seu código original continua aqui)
print("\n" + "="*80)
print("ANÁLISE DE CLUSTERS K-MEANS")

features = ["mean_hue", "mean_saturation", "mean_brightness",
            "contrast", "temperature", "sharpness", "edge_density", "entropy"]
X = result[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=100)
result["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
result["pca1"] = X_pca[:, 0]
result["pca2"] = X_pca[:, 1]

# Comparação com classes reais
conf_matrix = pd.crosstab(result['movie'], result['kmeans_cluster'])
print("Matriz de Confusão (Filme vs Cluster K-Means):")
print(conf_matrix)

ari = adjusted_rand_score(result['movie'], result['kmeans_cluster'])
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

# Salvar CSV final
result.to_csv("best_scenes_kmeans_pca_comparison.csv", index=False)
print(f"\n💾 CSV salvo: 'best_scenes_kmeans_pca_comparison.csv'")

# Mostrar estatísticas finais
print("\n" + "="*80)
print("ESTATÍSTICAS FINAIS")
print("="*80)
print(f"Total de frames processados: {len(result)}")
print(f"Número de clusters K-Means: {result['kmeans_cluster'].nunique()}")
print(f"Filmes analisados: {len(result['movie'].unique())}")
for movie in result['movie'].unique():
    count = len(result[result['movie'] == movie])
    print(f"  - {movie}: {count} frames")