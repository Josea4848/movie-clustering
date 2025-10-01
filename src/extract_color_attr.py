import cv2
import numpy as np
from matplotlib import pyplot as plt

# Função para extrair atributos de cor de uma imagem
def extract_color_features(img_path, show_hist=False):
    # Carrega imagem
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- 1. Média RGB ----
    mean_r = np.mean(img_rgb[:, :, 0])
    mean_g = np.mean(img_rgb[:, :, 1])
    mean_b = np.mean(img_rgb[:, :, 2])

    # ---- 2. Saturação média ----
    mean_saturation = np.mean(img_hsv[:, :, 1] / 255.0)  # normalizado 0-1

    # ---- 3. Luminosidade média ----
    mean_luminosity = np.mean(img_hsv[:, :, 2] / 255.0)  # normalizado 0-1

    # ---- 4. Contraste ----
    contrast = np.std(img_gray) / 255.0  # normalizado 0-1

    # ---- 5. Temperatura de cor aproximada ----
    # Simples: diferença média entre vermelho e azul
    temp_color = mean_r - mean_b  # positivo = quente, negativo = frio

    # ---- 6. Histograma simplificado (opcional) ----
    # 16 bins por canal
    hist_r = cv2.calcHist([img_rgb], [0], None, [16], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [16], [0, 256])
    hist_b = cv2.calcHist([img_rgb], [2], None, [16], [0, 256])
    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    hist = hist / hist.sum()  # normaliza

    # Mostrar histograma (opcional)
    if show_hist:
        plt.figure(figsize=(8,3))
        plt.title("Histograma RGB")
        plt.plot(hist_r, color='r')
        plt.plot(hist_g, color='g')
        plt.plot(hist_b, color='b')
        plt.show()

    # Retorna um dicionário com os atributos
    features = {
        'mean_r': mean_r,
        'mean_g': mean_g,
        'mean_b': mean_b,
        'mean_saturation': mean_saturation,
        'mean_luminosity': mean_luminosity,
        'contrast': contrast,
        'temp_color': temp_color,
        'histogram': hist
    }
    return features

# ---- EXEMPLO DE USO ----
img_path = '/home/enzoqua/Documents/projects/projeto-pdi/images/west.png'  # substitua pelo caminho da sua imagem
features = extract_color_features(img_path, show_hist=False)
print(features)
