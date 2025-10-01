import os
import sys
import cv2
import numpy as np
import pandas as pd
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def remove_black_bars(img, thresh=10):
    """
    Remove barras pretas da imagem.
    
    Args:
        img: imagem numpy array
        thresh: limiar para considerar "preto"
    
    Returns:
        imagem cortada sem barras pretas
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Soma ao longo das linhas (horizontal)
    row_sum = np.mean(gray, axis=1)
    top = np.argmax(row_sum > thresh)          # primeira linha não-preta
    bottom = len(row_sum) - np.argmax(row_sum[::-1] > thresh) - 1
    
    # Soma ao longo das colunas (vertical)
    col_sum = np.mean(gray, axis=0)
    left = np.argmax(col_sum > thresh)
    right = len(col_sum) - np.argmax(col_sum[::-1] > thresh) - 1
    
    # Verifica se encontrou áreas válidas para cortar
    if top < bottom and left < right:
        cortada = img[top:bottom, left:right]
        return cortada
    else:
        # Retorna a imagem original se não conseguir detectar bordas adequadas
        return img

def extract_scenes(video_path, output_dir="Output", threshold=27.0, min_duration=0.5, remove_bars=True):
    """
    Detecta cenas/shots em um vídeo e salva frames representativos + metadados em CSV.

    Args:
        video_path (str): caminho do vídeo de entrada.
        output_dir (str): pasta para salvar resultados.
        threshold (float): sensibilidade do detector de conteúdo (20–40 funciona bem).
        min_duration (float): duração mínima do shot em segundos.
        remove_bars (bool): se True, remove barras pretas dos frames.
    """
    subOut = os.path.basename(video_path).split('.')[0]
    output_dir = os.path.join(output_dir,subOut)

    os.makedirs(output_dir, exist_ok=True)

    # Inicializa PySceneDetect
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Processa o vídeo
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    print(f"[INFO] {len(scene_list)} cenas detectadas.")

    # Abrir vídeo no OpenCV para salvar frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    data = []
    for i, (start, end) in enumerate(scene_list):
        start_frame = start.get_frames()
        end_frame = end.get_frames()
        duration = (end_frame - start_frame) / fps

        if duration < min_duration:
            continue  # descarta cenas muito curtas

        # Escolhe 3 frames representativos (início, meio, fim)
        frames_to_save = [
            start_frame + int(0.1 * (end_frame - start_frame)),
            start_frame + int(0.5 * (end_frame - start_frame)),
            start_frame + int(0.9 * (end_frame - start_frame)),
        ]

        shot_id = f"shot_{i:03d}"
        shot_dir = os.path.join(output_dir, shot_id)
        os.makedirs(shot_dir, exist_ok=True)

        frames_saved_count = 0
        for j, fno in enumerate(frames_to_save):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if ret:
                # Remove barras pretas se solicitado
                if remove_bars:
                    try:
                        frame = remove_black_bars(frame, thresh=10)
                    except Exception as e:
                        print(f"[AVISO] Não foi possível remover barras do frame {fno}: {e}")
                        # Continua com frame original em caso de erro
                
                frame_path = os.path.join(shot_dir, f"frame_{j}.jpg")
                cv2.imwrite(frame_path, frame)
                frames_saved_count += 1

        data.append({
            "shot_id": shot_id,
            "start_time": start.get_timecode(),
            "end_time": end.get_timecode(),
            "duration_sec": duration,
            "frames_saved": frames_saved_count,
            "dir": shot_dir
        })

    cap.release()

    # Salva metadados em CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "scenes_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Metadados salvos em {csv_path}")

    return df


