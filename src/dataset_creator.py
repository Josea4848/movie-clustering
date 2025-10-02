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

  for root, dirs, files in os.walk(dir):
      for file in files:
        name, extension = os.path.splitext(file)
        frame_path = os.path.join(root, file)

        if extension == ".jpg":  
          try:
            data = extract_all(frame_path)
            data["movie"] = root.split("/")[1]  
            data["path"] = frame_path     
            df = pd.DataFrame([data])          
            df.to_csv(output_csv, mode="a", index=False, header=first)
            first = False
          except:
            print(print(os.path.join(root, file)))
    
if __name__ == "__main__":
  exportData()