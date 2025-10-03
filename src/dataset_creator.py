from AtributteExtractor import extract_all
import os
import pandas as pd
from AtributteExtractor import extract_all
from time import sleep

def readFrames(frames_dir):
  files = os.listdir(frames_dir)
  
def exportData():
  dir = "Output"
  output_csv = "frames_dataset.csv" 
  first = True
  df_curr = pd.read_csv(output_csv)

  for root, dirs, files in os.walk(dir):
      # Ignora o próprio dir
      if root == dir:      
        continue
      
      movie = root.split("/")[1]
      exists = (df_curr["movie"] == movie).any()
      
      # Ignora o filme, caso já esteja cadastrado
      if exists:
        continue

      for file in files:
        name, extension = os.path.splitext(file)
        frame_path = os.path.join(root, file)

        if extension == ".jpg":  
          try:
            data = extract_all(frame_path)
            data["movie"] = movie
            data["path"] = frame_path     
            df = pd.DataFrame([data])          
            df.to_csv(output_csv, mode="a", index=False, header=first)
            first = False
          except:
            print(print(os.path.join(root, file)))
    
if __name__ == "__main__":
  exportData()