import cv2
import numpy as np
from skimage.measure import shannon_entropy

def extract_visual_features(img_path) -> dict[str, float]:
    """
    Extrai atributos visuais de uma imagem.
    
    Args:
        img_path (str): path da imagem a ser processada.

    Returns:
        dict[str, float]: Dicionário com valores para os atributos
    """


    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    img = img.astype(np.float32) / 255.0

    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.float32) / 179.0
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0

    # MEDIA HSV
    mean_hue = np.mean(H)
    mean_sat = np.mean(S)
    mean_bri = np.mean(V)

    # CONTRASTE
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)

    # TEMPERATURA MÉDIA
    R = img[..., 2]
    B = img[..., 0]
    temperature = np.mean(R) - np.mean(B)

    # NITIDEZ
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # DENSIDADE DE BORDAS
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # ENTROPIA
    entropy = shannon_entropy(gray)

    features = {
        'mean_hue': float(mean_hue),
        'mean_saturation': float(mean_sat),
        'mean_brightness': float(mean_bri),
        'contrast': float(contrast),
        'temperature': float(temperature),
        'sharpness': float(lap_var),
        'edge_density': float(edge_density),
        'entropy': float(entropy)
    }

    return features

