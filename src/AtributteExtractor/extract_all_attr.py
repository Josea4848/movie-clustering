import cv2
import numpy as np
from skimage.measure import shannon_entropy

def extract_visual_features(img_path, k=3):

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

    # # MATIZ DOMINANTE
    # mask = S > 0.2
    # h_vals = H[mask].reshape(-1, 1)
    # if len(h_vals) > 0:
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    #     compactness, labels, centers = cv2.kmeans(h_vals.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #     # MAIOR CLUSTER
    #     counts = np.bincount(labels.flatten())
    #     dominant_hue = centers[np.argmax(counts)][0]
    # else:
    #     dominant_hue = float('nan')

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
        #'dominant_hue': float(dominant_hue),
        'edge_density': float(edge_density),
        'entropy': float(entropy)
    }

    return features

