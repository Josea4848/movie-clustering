from AtributteExtractor import extract_all
import os
import pandas as pd
from time import sleep

def exportData():
    dir = "Output"
    output_csv = "frames_dataset.csv"

    # Verifica se o CSV já existe
    if os.path.exists(output_csv):
        df_curr = pd.read_csv(output_csv)
        first = False
    else:
        df_curr = pd.DataFrame()
        first = True

    for root, dirs, files in os.walk(dir):
        # Ignora o próprio diretório principal
        if root == dir:
            continue
        
        movie = root.split("/")[1]
        exists = (df_curr["movie"] == movie).any() if not df_curr.empty else False
        
        # Ignora o filme se já estiver cadastrado
        if exists:
            continue

        for file in files:
            name, extension = os.path.splitext(file)
            frame_path = os.path.join(root, file)

            if extension.lower() == ".jpg":  
                try:
                    data = extract_all(frame_path)
                    data["movie"] = movie
                    data["path"] = frame_path     
                    df = pd.DataFrame([data])          
                    df.to_csv(output_csv, mode="a", index=False, header=first)
                    first = False
                except Exception as e:
                    print(f"Erro ao processar {frame_path}: {e}")

if __name__ == "__main__":
    exportData()
