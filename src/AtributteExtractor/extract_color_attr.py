import cv2
import numpy as np
from matplotlib import pyplot as plt


# ---- Função que você já criou ----
def extract_color_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None  # segurança: caso não consiga abrir
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Média RGB
    mean_r = np.mean(img_rgb[:, :, 0])
    mean_g = np.mean(img_rgb[:, :, 1])
    mean_b = np.mean(img_rgb[:, :, 2])

    # 2. Saturdação média
    mean_saturation = np.mean(img_hsv[:, :, 1] / 255.0)

    # 3. Luminosidade média
    mean_luminosity = np.mean(img_hsv[:, :, 2] / 255.0)

    # 4. Contraste
    contrast = np.std(img_gray) / 255.0

    # 5. Temperatura de cor (aprox.)
    temp_color = mean_r - mean_b

    features = {
        'mean_r': mean_r,
        'mean_g': mean_g,
        'mean_b': mean_b,
        'mean_saturation': mean_saturation,
        'mean_luminosity': mean_luminosity,
        'contrast': contrast,
        'temp_color': temp_color,
    }
    return features
