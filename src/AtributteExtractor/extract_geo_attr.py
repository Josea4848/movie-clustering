import cv2
import numpy as np

def extract_geo_features(img_path: str):
    # --- Carregar imagem em escala de cinza ---
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Erro ao carregar imagem")

    h, w = img.shape

    # =============================
    # 1. Bordas e Orientações
    # =============================
    edges = cv2.Canny(img, 100, 200)

    # Gradientes de Sobel (x = horizontal, y = vertical)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Ângulo da borda
    angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angles = angles[edges > 0]  # apenas onde há borda

    # Classificar em faixas de orientação
    horizontais = np.sum((angles >= -22.5) & (angles < 22.5)) + \
                  np.sum((angles >= 157.5) | (angles < -157.5))
    verticais   = np.sum((angles >= 67.5) & (angles < 112.5)) + \
                  np.sum((angles <= -67.5) & (angles > -112.5))
    diagonais   = len(angles) - horizontais - verticais

    total = max(len(angles), 1)  # evitar divisão por zero
    proporcao_h = horizontais / total
    proporcao_v = verticais / total
    proporcao_d = diagonais / total

    # =============================
    # 2. Simetria (esquerda vs direita)
    # =============================
    metade_esq = img[:, :w//2]
    metade_dir = cv2.flip(img[:, w//2:], 1)  # espelha metade direita
    min_w = min(metade_esq.shape[1], metade_dir.shape[1])

    dif = np.mean(np.abs(metade_esq[:, :min_w].astype(float) -
                         metade_dir[:, :min_w].astype(float)))
    simetria = 1 - dif / 255.0  # normalizado (0 = nada simétrico, 1 = idêntico)

    # =============================
    # 3. Centro de Massa da Intensidade
    # =============================
    y_coords, x_coords = np.indices(img.shape)
    total_intensidade = np.sum(img)
    if total_intensidade == 0:
        cx, cy = w/2, h/2  # fallback = centro da imagem
    else:
        cx = np.sum(x_coords * img) / total_intensidade
        cy = np.sum(y_coords * img) / total_intensidade

    # Normalizar [0,1]
    centro_x_norm = cx / w
    centro_y_norm = cy / h

    # =============================
    # Resultado
    # =============================
    atributos = {
        "proporcao_bordas_horizontais": proporcao_h,
        "proporcao_bordas_verticais": proporcao_v,
        "proporcao_bordas_diagonais": proporcao_d,
        "simetria_esq_dir": simetria,
        "centro_luz_x": centro_x_norm,
        "centro_luz_y": centro_y_norm
    }

    return atributos


