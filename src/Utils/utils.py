import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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
        movie_frames = result_df[result_df['movie'] == movie]
        
        # Pegar os primeiros N frames para visualização
        sample_frames = movie_frames.head(num_frames_per_movie)
        
        fig, axes = plt.subplots(1, len(sample_frames), figsize=(18, 4))
        if len(sample_frames) == 1:
            axes = [axes]
        
        for idx, (_, row) in enumerate(sample_frames.iterrows()):
            img_path = row['path']
            
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
    for movie in result_df['movie'].unique():
        movie_frames = result_df[result_df['movie'] == movie]
        print(f"{movie} ({len(movie_frames)} frames):")
        
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
