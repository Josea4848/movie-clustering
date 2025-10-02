from FrameExtractor import extract_scenes 
import os
import sys 

if len(sys.argv) < 2:
  raise Exception("Quantidade de argumentos passados inválida")

if not os.path.isdir(sys.argv[1]):
  raise FileNotFoundError("Diretório inválido")

movies_dir = sys.argv[1]
movies = os.listdir(movies_dir)

for movie in movies:
  try:
    movie_path = os.path.join(movies_dir, movie)

    print(f"Extraindo cenas {movie}")
    result = extract_scenes(movie_path)
  except:
    print(f"[{movie}] Erro ao processar cenas do filme")

