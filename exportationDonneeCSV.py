from regleDeveloppeCouche import *
from regleSouleveDeTerre import * 
from squatDataRetrieve import *
import pandas as pd
import os

indice = None

def configVarChemin(mouvement : IndiceMouvement) -> str:
    """Parcours le fichier configExportationCSV pour récupérer la bonne valeur pour la construction du chemin d'enregistrement.
    Args:
        mouvement (IndiceMouvement): Mouvement effectué.
    Returns:
        str: Retourne la valeur nécessaire pour la structuration du chemin."""
    filename = "configExportationCSV.txt"
    with open(filename, "r") as file:
        for line in file:
            line = line.split("#")[0].strip()
            parts = line.split(" : ")
            if len(parts) == 2 and parts[0].strip() == str(mouvement):
                return parts[1].strip()
    return None  # Retourne None si l'index n'est pas trouvé

def incrementerIndice(mouvement : IndiceMouvement):
    """Incrémente l'indice du mouvement dans le fichier configExportationCSV.
    Args:
        mouvement (IndiceMouvement): Mouvement effectué."""
    global indice
    filename = "configExportationCSV.txt"
    with open(filename, "r") as file:
        lines = file.readlines()
    
    with open(filename, "w") as file:
        for line in lines:
            parts = line.split("#")  # Séparer le commentaire
            data = parts[0].strip().split(" : ")
            if len(data) == 2 and data[0].strip() == str(mouvement):
                new_value = int(data[1].strip()) + 1  
                file.write(f"{data[0]} : {new_value} #" + parts[1])
            else:
                file.write(line)
    indice = None

def structurationChemin(pathname : str, mouvement : IndiceMouvement) -> tuple:
    """Créer le chemin pour savegarder le fichier en fonction du mouvement effectué.
    Args:
        filename (str): Nom du fichier.
        mouvement (IndiceMouvement): Mouvement effectué.
    Returns:
        tuple: Retourne le chemin ainsi que si le fichier doit être créer ou non."""
    global indice
    chemin, creation = None, None
    if indice is not None:
        var = indice
    else:
        indice, var = configVarChemin(mouvement), indice
    if var != None:
        if mouvement == IndiceMouvement.SQUAT.value:
            chemin = pathname + "/SQUAT"
            os.makedirs(chemin, exist_ok=True)
            chemin += "/squat" + var + ".csv"
        elif mouvement == IndiceMouvement.DEADLIFT.value:
            chemin = pathname + "/DEADLIFT"
            os.makedirs(chemin, exist_ok=True)
            chemin += "/deadlift" + var + ".csv"
        elif mouvement == IndiceMouvement.BENCH.value:
            chemin = pathname + "/BENCH"
            os.makedirs(chemin, exist_ok=True)
            chemin += "/bench" + var + ".csv"
        else:
            print("Mouvement inconnu : Impossibe d'exporter les données en csv")
            return None, False
    else:
        print("Erreur lors de la récupération de l'indice")
        return None, False
    if os.path.exists(chemin):
        creation = False
    else:
        creation = True
    return chemin, creation

def recupDataSquat(touteCoordonnee : list) -> str:
    """Créer la structre pour les données de squat.
    Args:
        touteCoordonnee (list): Regroupe toutes les coordonnées depuis le début de la vidéo.
    Returns:
        str: Chaîne contenant les données."""
    data = traitementSquat(touteCoordonnee) 
    dataAngle = data[1]
    data = data[0][0]
    if len(data) >= 7:
        dataFichier = {
            "Epaule gauche :": [data[0]],
            "Epaule droite :": [data[4]],
            "Hanche gauche :": [data[1]],
            "Hanche droite :": [data[5]],
            "Genoux gauche :": [data[2]],
            "Genoux droit :": [data[6]],
            "Talons gauche :": [data[3]],
            "Angle gauche :": [dataAngle[0]],
            "Angle droit :": [dataAngle[1]]
        }
    else:
        print("Les données de l'articulation sont incomplètes.")
        dataFichier = {}

    return dataFichier

def recupDataDeadlift(touteCoordonnee : list) -> str:
    """Créer la structre pour les données de deadlift.
    Args:
        touteCoordonnee (list): Regroupe toutes les coordonnées depuis le début de la vidéo.
    Returns:
        str: Chaîne contenant les données."""
    data = souleverDeTerreValue(touteCoordonnee[-1])
    dataFichier = {
        "Angle genoux gauche :":[data[0][1]],
        "Angle genoux droit :":[data[0][0]],
        "Epaule gauche :":[data[1][1]],
        "Epaule droite :":[data[1][0]],
        "Hanche gauche :":[data[1][3]],
        "Hanche droite :":[data[1][2]],
        "Point gauche barre :":[data[2][0]],
        "Point droit barre :":[data[2][1]],
        "Poignet droit :":[data[5][3]],
        "Poignet gauche :":[data[5][2]],
        "Talon gauche :":[data[6][1]],
        "Talong droit :":[data[6][0]]
    }
    return dataFichier

def recupDataBench(touteCoordonnee : list) -> str:
    """Créer la structre pour les données de bench.
    Args:
        touteCoordonnee (list): Regroupe toutes les coordonnées depuis le début de la vidéo.
    Returns:
        str: Chaîne contenant les données."""
    data = developpeCoucheCoord(touteCoordonnee[-1])
    pointGauche, pointDroit = data[5][0]
    dataFichier = {
        "Angle Tête :":[data[0]],
        "Epaule gauche :":[data[1][1]],
        "Epaule droite :":[data[1][0]],
        "Fessier gauche :":[data[1][3]],
        "Fessier droit :":[data[1][2]],
        "Talon gauche :":[data[2][1]],
        "Talon droit :":[data[2][0]],
        "Point gauche barre :":[pointGauche],
        "Point droit barre :":[pointDroit],
        "Coude gauche :":[data[5][3]],
        "Coude droit :":[data[5][4]],
        "Hanche gauche :":[data[5][5]],
        "Hanche gauche :":[data[5][6]],
        "Poitrine :":[data[5][7]],
    }
    return dataFichier

def structurationData(touteCoordonnee : list, mouvement : IndiceMouvement) -> str:
    """Créer la structre pour l'exportation des données en csv en fonction du mouvement.
    Args:
        touteCoordonnee (list): Regroupe toutes les coordonnées depuis le début de la vidéo.
    Returns:
        str: Chaîne contenant les données."""
    if mouvement == IndiceMouvement.SQUAT.value:
        dataFichier = recupDataSquat(touteCoordonnee)
    elif mouvement == IndiceMouvement.DEADLIFT.value:
        dataFichier = recupDataDeadlift(touteCoordonnee)
    elif mouvement == IndiceMouvement.BENCH.value:
        dataFichier = recupDataBench(touteCoordonnee)
    else:
        print("Mouvement inconnu : Impossibe d'exporter les données en csv")
        return {}
    return dataFichier

def exportationCSV(touteCoordonnee : list, pathname : str, mouvement : IndiceMouvement) -> None:
    """Permet d'exporter les données d'une frame dans un fichier csv en focntion du mouvement.
    Args:
        touteCoordonnee (list): Regroupe toutes les coordonnées depuis le début de la vidéo.
        pathname (str) : Chemin sélectionné
        mouvement (IndiceMouvement) : Mouvement effectué"""
    if touteCoordonnee[-1][0] != [] or touteCoordonnee[-1][1] != []:
        chemin, creation = structurationChemin(pathname, mouvement) # chemin et (True s'il faut creer, False si on ajoute) 
        data = structurationData(touteCoordonnee, mouvement)
        df = pd.DataFrame(data)
        if chemin != None:
            if creation and data != {}:
                df.to_csv(chemin)
            elif not creation and data != {}:
                df.to_csv(chemin, mode='a', header=False)
    else:
        print("Impossible d'enregistrer les coordonées pour cette frame")