import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Caminho para o CSV
csv_path = "frames_dataset.csv"   # troque pelo nome do seu arquivo

# Ler CSV
df = pd.read_csv(csv_path, header=0)

# Garantir que colunas sejam numéricas (exceto 'movie')
for col in df.columns:
    if col != "movie":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Separar atributos numéricos
X = df.drop(columns=["movie"])
X = X.drop(columns=["path"])
X = X.dropna()

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para 2 dimensões
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Criar DataFrame com resultado
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["movie"] = df["movie"]

# Plot colorido por filme
plt.figure(figsize=(8,6))
movies = df_pca["movie"].unique()  # lista de filmes únicos
for movie in movies:
    subset = df_pca[df_pca["movie"] == movie]
    plt.scatter(subset["PC1"], subset["PC2"], alpha=0.7, label=movie)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Redução para 2D")
plt.legend()
plt.grid(True)
plt.show()
