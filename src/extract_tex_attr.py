import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

def extract_texture_features(image_path):
    # Carregar imagem
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalizar para 8 bits
    gray_8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 1. GLCM (Gray Level Co-occurrence Matrix) ---
    # Distâncias [1, 2, 4] pixels e ângulos [0, 45, 90, 135 graus]
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    # Propriedades comuns
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    # --- 2. Nitidez (Laplaciano) ---
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # --- 3. Entropia ---
    entropy = shannon_entropy(gray)

    features = {
        "GLCM_contrast": contrast,
        "GLCM_homogeneity": homogeneity,
        "GLCM_energy": energy,
        "GLCM_correlation": correlation,
        "Laplacian_variance": lap_var,
        "Entropy": entropy
    }

    print("Aaa")
    return features

# Exemplo de uso
features = extract_texture_features("/home/enzoqua/Documents/projects/projeto-pdi/images/fram2_2_cropped.png")
print(features)
