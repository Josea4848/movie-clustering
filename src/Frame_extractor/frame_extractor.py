import os
import sys
import cv2
import pandas as pd
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def extract_scenes(video_path, output_dir="output_scenes", threshold=27.0, min_duration=0.5):
    """
    Detecta cenas/shots em um vídeo e salva frames representativos + metadados em CSV.

    Args:
        video_path (str): caminho do vídeo de entrada.
        output_dir (str): pasta para salvar resultados.
        threshold (float): sensibilidade do detector de conteúdo (20–40 funciona bem).
        min_duration (float): duração mínima do shot em segundos.
    """
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

        for j, fno in enumerate(frames_to_save):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(shot_dir, f"frame_{j}.jpg")
                cv2.imwrite(frame_path, frame)

        data.append({
            "shot_id": shot_id,
            "start_time": start.get_timecode(),
            "end_time": end.get_timecode(),
            "duration_sec": duration,
            "frames_saved": len(frames_to_save),
            "dir": shot_dir
        })

    cap.release()

    # Salva metadados em CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "scenes_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Metadados salvos em {csv_path}")

    return df


if __name__ == "__main__":
    
    if len(sys.argv)  < 2:
      print("Informe o caminho para armazenamento dos frames")
      sys.exit(0)

    video_file = sys.argv[1] # coloque aqui seu vídeo curto
    output_dir =  sys.argv[2] # caminho para pasta que armazena os frames
    
    df = extract_scenes(video_file, output_dir=output_dir, threshold=27.0)
    print(df.head())