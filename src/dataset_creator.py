from AtributteExtractor import extract_visual_features
import os
import pandas as pd
from time import sleep

def getDirectorMap() -> dict[str, str]:
    """
    Retorna um dicionário para mapeamento do diretor para cada filme

    Returns:
        dict[str, str]: mapa diretor para filme
    """

    # Mapeamento manual dos diretores
    movie_to_director = {
        "Batman1989": "Tim Burton",
        "BatmanReturns1992": "Tim Burton",
        "SleepyHollow1999": "Tim Burton",
        "AllAboutMyMother1999": "Pedro Almodóvar",
        "Volver2006": "Pedro Almodóvar",
        "WomenOnTheVergeOfANervousBreakdown1988": "Pedro Almodóvar",
        "TheTreeofLife2011": "Terrence Malick",
        "KnightOfCups2015": "Terrence Malick",
        "AHiddenLife2019": "Terrence Malick",
        "20462004": "Wong Kar-Wai",
        "InTheMoodForLove2000": "Wong Kar-Wai",
        "DaysOfBeingWild1990": "Wong Kar-Wai",
        "RhapsodyInAugust1991": "Akira Kurosawa",
        "Dreams1990": "Akira Kurosawa",
        "Madadayo1993": "Akira Kurosawa",
        "EliteSquad2007": "José Padilha",
        "EliteSquad2TheEnemyWithin2010": "José Padilha",
        "TheBirds1963": "Alfred Hitchcock",
        "NorthByNorthwest1959": "Alfred Hitchcock",
        "OnceUponaTimeintheWest1968": "Sergio Leone", 
        "MadMaxFuryRoad2015": "George Miller"
    }

    return movie_to_director



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
            print(f"Filme ignorado: {movie}")
            continue

        for file in files:
            name, extension = os.path.splitext(file)
            frame_path = os.path.join(root, file)

            if extension.lower() == ".jpg":  
                try:
                    data = extract_visual_features(frame_path)
                    data["movie"] = movie
                    data["path"] = frame_path     
                    df = pd.DataFrame([data])          
                    df.to_csv(output_csv, mode="a", index=False, header=first)
                    first = False
                except Exception as e:
                    print(f"Erro ao processar {frame_path}: {e}")

    # Mapea diretores para cada filme
    df = pd.read_csv(output_csv)
    df["director"] = df["movie"].map(getDirectorMap())

if __name__ == "__main__":
    exportData()
