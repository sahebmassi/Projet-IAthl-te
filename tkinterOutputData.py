from traitementVideo import *
from trouverArticulationYolo import *
import mimetypes
from tkinter import messagebox
import os

correspondanceTraitementVideo = {
    "" : None,
    "rien" : rien,
    "boxes" : affBox,
    "articulation" : affArt,
    "articulation et boxes" : affArtBox,
    "Developpe Coucher" : affDeveloppeCouche,
    "Squat" : affichageOrienteSquat,
    "Soulever Terre" : affInfoSouleveTerre
}

correspondanceTrouverArticulation = {
    "" : None, 
    "yolo" : yoloVOT,
    "yolo boxe centree" : yoloCentreBox,
    "yolo boxe souris" : yoloSourisBox,
    "yolo proche barre" : yoloProcheBarreArtBox,
}

def choixMaterielOutput(choixMateriel : str) -> bool :
    """Rends un booléen pour savoir si un GPU peut-être utilisé
    Args:
        choixMateriel (str): choix de l'interface
    Returns:
        bool: booléen pour l'utilisation d'un GPU"""
    return not (choixMateriel[:10] == "Plateforme")

def choixTraitementVideoOutput(choixTraitementVideo : str) -> any :
    """Rends la correspondance entre le choix de traitement de vidéo et le code 
    Args:
        choixTraitementVideo (str): le choix de l'utilisateur
    Returns:
        any: la fonction qui est a appliqué pour le traitement de vidéo"""
    return correspondanceTraitementVideo[choixTraitementVideo]

def choixTrouverArticulationOutput(choixTrouverArticulation : str) -> any :
    """Rends la correspondance entre le choix de trouver les articulations et le code 
    Args:
        choixTraitementVideo (str): le choix de l'utilisateur
    Returns:
        any: la fonction qui est a appliqué pour trouver les articulations"""
    return correspondanceTrouverArticulation[choixTrouverArticulation]

def estVideo(nom_fichier : str) -> bool:
    """test si le nom de fichier donné est un format vidéo
    Args:
        nom_fichier (str): le nom du fichier
    Returns:
        bool: booléen de si le fichier est un format vidéo"""
    mimeType, _ = mimetypes.guess_type(nom_fichier)
    return mimeType is not None and mimeType.startswith("video")

def verifierInformation(traitementVideo : any, trouverArticulation : any, enregistrementVideoAvant : str, enregistrementVideoApres : str, enregistrementJson : str) -> bool :
    """vérifie si il n'y a pas d'erreur dans les entrées de l'utilisateur
    Args:
        traitementVideo (any): choix de l'utilisateur traiter les données
        trouverArticulation (any): choix de l'utilisateur trouver les articulations
        enregistrementVideoAvant (str): choix de l'utilisateur pour enregistrer la vidéo avant traitement
        enregistrementVideoApres (str): choix de l'utilisateur pour enregistrer la vidéo après traitement
        enregistrementJson (str): choix de l'utilisateur pour enregistrer les données dans un fichier Json
    Returns:
        bool: retourne si l'interface peut se lancer (si il n'y a pas d'erreurs)"""
    if traitementVideo == None :
        messagebox.showerror("Erreur", "Veuillez choisir un traitement pour la vidéo.")
        return False
    if trouverArticulation == None :
        messagebox.showerror("Erreur", "Veuillez choisir un moyen de trouver les articulations pour la vidéo.")
        return False
    if enregistrementVideoAvant :
        if not estVideo(enregistrementVideoAvant) :
            messagebox.showerror("Erreur", "Veuillez choisir de créer un fichier format vidéo.")
            return False
    if enregistrementVideoApres :
        if not estVideo(enregistrementVideoApres) :
            messagebox.showerror("Erreur", "Veuillez choisir de créer un fichier format vidéo.")
            return False
    if enregistrementJson :
        if not os.path.isdir(enregistrementJson):
            messagebox.showerror("Erreur", "Veuillez choisir un dossier pour enregistrer les données.")
            return False
    return True
