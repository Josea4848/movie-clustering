import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Carregar dados
# -------------------------------
csv_path = "frames_dataset.csv"
df = pd.read_csv(csv_path, header=0)

# -------------------------------
# 2. Selecionar colunas numéricas
# -------------------------------
features = ["mean_r", "mean_g", "mean_b", "mean_saturation",
            "mean_luminosity", "contrast", "temp_color", "Laplacian_variance"]

# -------------------------------
# 3. Remover linhas não numéricas
# -------------------------------
for col in features:
    df = df[pd.to_numeric(df[col], errors='coerce').notna()]

# -------------------------------
# 4. Converter para float
# -------------------------------
for col in features:
    df[col] = df[col].astype(float)

X = df[features].values

# -------------------------------
# 5. Normalizar dados
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 6. DBSCAN no espaço original
# -------------------------------
dbscan = DBSCAN(eps=2.0, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
df['cluster'] = clusters

# -------------------------------
# 7. PCA apenas para visualização
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -------------------------------
# 8. Visualização
# -------------------------------
plt.figure(figsize=(10,7))
palette = sns.color_palette("hsv", len(set(clusters)))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette=palette, legend='full')
plt.title("DBSCAN clustering das cenas (PCA 2D apenas para visualização)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# -------------------------------
# 9. Análise básica
# -------------------------------
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print("Número de clusters encontrados (excluindo outliers):", num_clusters)
print(df.groupby('cluster')['movie'].value_counts())
