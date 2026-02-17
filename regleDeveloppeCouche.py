from enumIndice import *
from utils import calculerAngle

def orientationTete(coordonnees : list) -> list:
    """Permet d'avoir les coordonnées pour vérifier l'orientation de la tête
    Args:
        coordonnees (list): les coordonées de la dernière frame
    Returns:
        list: les coordonées pour vérifier la partie du mouvement"""
    if len(coordonnees) > 0:
        coordonnees = coordonnees[0]
        return calculerAngle(coordonnees, IndiceYolo.OREILLE_GAUCHE.value, IndiceYolo.NEZ.value, IndiceYolo.OREILLE_DROITE.value)
    else:
        return []

def contactBanc(coordonnees : list) -> list:
    """Permet d'avoir les coordonnées pour vérifier le contact avec le banc
    Args:
        coordonnees (list): les coordonées de la dernière frame
    Returns:
        list: les coordonées pour vérifier la partie du mouvement"""
    if len(coordonnees) > 0:
        coordonnees = coordonnees[0]
        epaule_droite = coordonnees[IndiceYolo.EPAULE_DROITE.value]
        epaule_gauche = coordonnees[IndiceYolo.EPAULE_GAUCHE.value]
        fessier_droit = []
        fessier_gauche = []
        return [epaule_droite,epaule_gauche,fessier_droit,fessier_gauche]
    else:
        return []

def piedPlat(coordonnees : list) -> list:
    """Permet d'avoir les coordonnées pour vérifier si les pieds sont plats
    Args:
        coordonnees (list): les coordonées de la dernière frame
    Returns:
        list: les coordonées pour vérifier la partie du mouvement"""
    if len(coordonnees) > 0:
        coordonnees = coordonnees[0]
        return [coordonnees[IndiceYolo.TALON_DROIT.value], coordonnees[IndiceYolo.TALON_GAUCHE.value]]
    else:
        return []
    
def mainSerrer(coordonnees : list) -> list:
    """Permet d'avoir les coordonnées pour vérifier si les mains sont serré
    Args:
        coordonnees (list): les coordonées de la dernière frame
    Returns:
        list: les coordonées pour vérifier la partie du mouvement"""
    return []

def espacementMain(coordonnees : list) -> list:
    """Permet d'avoir les coordonnées pour vérifier l'espacement des mains
    Args:
        coordonnees (list): les coordonées de la dernière frame
    Returns:
        list: les coordonées pour vérifier la partie du mouvement"""
    return []

def descenteBarre(coordonnees : list) -> list:
    """Permet d'avoir les coordonnées pour vérifier la descente de la barre
    Args:
        coordonnees (list): les coordonées de la dernière frame
    Returns:
        list: les coordonées pour vérifier la partie du mouvement"""
    if  len(coordonnees[1]) > 0 and len(coordonnees[0]) > 0:
        xMin, yMin, xMax, yMax = coordonnees[1][0]
        barre = [(xMin,(yMin + yMax)//2),(xMax,(yMin + yMax)//2)]
        epaule_gauche = coordonnees[0][0][IndiceYolo.EPAULE_GAUCHE.value]
        epaule_droite = coordonnees[0][0][IndiceYolo.EPAULE_DROITE.value]
        coude_gauche = coordonnees[0][0][IndiceYolo.COUDE_GAUCHE.value]
        coude_droite = coordonnees[0][0][IndiceYolo.COUDE_DROIT.value]

        hanche_gauche = coordonnees[0][0][IndiceYolo.HANCHE_GAUCHE.value]
        hanche_droite = coordonnees[0][0][IndiceYolo.HANCHE_DROITE.value]
        poitrine_abdo = ((epaule_gauche[0] + epaule_droite[0] + hanche_gauche[0] + hanche_droite[0]) / 4, (epaule_gauche[1] + epaule_droite[1] + hanche_gauche[1] + hanche_droite[1]) / 4)

        return [barre, epaule_gauche, epaule_droite, coude_gauche, coude_droite, hanche_gauche, hanche_droite, poitrine_abdo]
    else :
        return []

def developpeCoucheCoord(toutesCoordonnees  : list) -> bool:
    """Permet d'avoir les coordonnées pour vérifier le mouvement en entier
    Args:
        toutesCoordonnees (list): les coordonées depuis le début du mouvement
    Returns:
        list: les coordonées pour vérifier le mouvement"""
    orientationTeteVal = orientationTete(toutesCoordonnees[0])
    contactBancVal = contactBanc(toutesCoordonnees[0])
    piedPlatVal = piedPlat(toutesCoordonnees[0])
    mainSerrerVal = mainSerrer(toutesCoordonnees)
    espacementMainVal = espacementMain(toutesCoordonnees)
    descenteBarreVal = descenteBarre(toutesCoordonnees)
    return [orientationTeteVal,contactBancVal,piedPlatVal,mainSerrerVal,espacementMainVal,descenteBarreVal]

