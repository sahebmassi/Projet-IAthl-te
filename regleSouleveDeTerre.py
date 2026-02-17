from utils import *
from enumIndice import *

def verouillerGenoux(coordonnees : list) -> list :
    """retourne les valeur pour savoir si le genoux est vérouillé
    Args:
        coordonnees (list): coordonnées de la dernière frame
    Returns:
        list: les coordonées pour vérifier cette partie du mouvement"""
    if len(coordonnees) > 0 :
        coordonnees = coordonnees[0]
        return calculerAngle(coordonnees,IndiceYolo.HANCHE_DROITE.value,IndiceYolo.GENOU_DROIT.value,IndiceYolo.TALON_DROIT.value),calculerAngle(coordonnees,IndiceYolo.HANCHE_GAUCHE.value,IndiceYolo.GENOU_GAUCHE.value,IndiceYolo.TALON_GAUCHE.value)
    else :
        return []

def epaulesArriere(coordonnees : list) -> list :
    """retourne les valeur pour savoir si les épaules sont en arrières
    Args:
        coordonnees (list): coordonnées de la dernière frame
    Returns:
        list: les coordonées pour vérifier cette partie du mouvement"""
    if len(coordonnees) > 0 :
        coordonnees = coordonnees[0]
        return [coordonnees[IndiceYolo.EPAULE_DROITE.value],coordonnees[IndiceYolo.EPAULE_GAUCHE.value],coordonnees[IndiceYolo.HANCHE_DROITE.value],coordonnees[IndiceYolo.HANCHE_GAUCHE.value]]
    else :
        return []

def barreVersBas(coordonnees : list) -> list :
    """retourne les valeur pour savoir la barre est descendu vers le bas
    Args:
        coordonnees (list): coordonnées de la dernière frame
    Returns:
        list: les coordonées pour vérifier cette partie du mouvement"""
    if  len(coordonnees[1]) > 0 and len(coordonnees[0]) > 0:
        xMin, yMin, xMax, yMax = coordonnees[1][0]
        barre = [(xMin,(yMin + yMax)//2),(xMax,(yMin + yMax)//2)]
        epaules = [coordonnees[0][0][IndiceYolo.EPAULE_DROITE.value], coordonnees[0][0][IndiceYolo.EPAULE_GAUCHE.value]]
        barre.extend(epaules)
        return barre
    else :
        return []
    
def aiderCuisse(coordonnees : list) -> list :
    """retourne les valeur pour savoir si les cuisses ont aidées pour monter la barre
    Args:
        coordonnees (list): coordonnées de la dernière frame
    Returns:
        list: les coordonées pour vérifier cette partie du mouvement"""
    if  len(coordonnees[1]) > 0 and len(coordonnees[0]) > 0:
        xMin, yMin, xMax, yMax = coordonnees[1][0]
        barre = [(xMin,(yMin + yMax)//2),(xMax,(yMin + yMax)//2)]
        cuisses = [coordonnees[0][0][IndiceYolo.HANCHE_DROITE.value], coordonnees[0][0][IndiceYolo.HANCHE_GAUCHE.value],coordonnees[0][0][IndiceYolo.GENOU_DROIT.value], coordonnees[0][0][IndiceYolo.GENOU_GAUCHE.value]]
        barre.extend(cuisses)
        return barre
    else :
        return []

def descendreBarre(coordonnees : list) -> list :
    """retourne les valeur pour savoir si la barre descend bien à partir de l'autorisation
    Args:
        coordonnees (list): coordonnées de la dernière frame
    Returns:
        list: les coordonées pour vérifier cette partie du mouvement"""
    if  len(coordonnees[1]) > 0:
        xMin, yMin, xMax, yMax = coordonnees[1][0]
        barre = [(xMin,(yMin + yMax)//2),(xMax,(yMin + yMax)//2)]
        return barre
    else :
        return []

def ramenerBarreControler(coordonnees : list) -> list :
    """retourne les valeur pour savoir si le genoux est vérouillé
    Args:
        coordonnees (list): coordonnées de la dernière frame
    Returns:
        list: les coordonées pour vérifier cette partie du mouvement"""
    if  len(coordonnees[1]) > 0 and len(coordonnees[0]) > 0:
        xMin, yMin, xMax, yMax = coordonnees[1][0]
        barre = [(xMin,(yMin + yMax)//2),(xMax,(yMin + yMax)//2)]
        epaules = [coordonnees[0][0][IndiceYolo.POIGNET_DROIT.value], coordonnees[0][0][IndiceYolo.POIGNET_GAUCHE.value]]
        barre.extend(epaules)
        return barre
    else :
        return []

def bougerPieds(coordonnees : list) -> list:
    """retourne les valeur pour savoir si les pieds on bougés
    Args:
        coordonnees (list): coordonnées de la dernière frame
    Returns:
        list: les coordonées pour vérifier cette partie du mouvement"""
    if len(coordonnees) > 0 :
        return [coordonnees[0][IndiceYolo.TALON_DROIT.value], coordonnees[0][IndiceYolo.TALON_GAUCHE.value]]
    else :
        return []

def souleverDeTerreValue(toutesCoordonnees : list) -> list: # Cette fonction fonctionne que pour une personne et une box de barre
    """retourne les valeur pour savoir si le mouvement est bien fais
    Args:
        toutesCoordonnees (list): coordonnées depuis le début du mouvement
    Returns:
        list: les coordonées pour vérifier le mouvement"""
    verouilletGenouxValue = verouillerGenoux(toutesCoordonnees[0]) #[angle genoux droit, angle genoux gauche]
    epaulesArriereValue = epaulesArriere(toutesCoordonnees[0]) #[coord epaule droite, coord epaule gauche, coord hanche droite, coord hanche gauche]
    barreVersBasValue = barreVersBas(toutesCoordonnees) #[point gauche de la barre, point droit de la barre, coord epaules droite, coord epaules gauche]
    aiderCuisseValue = aiderCuisse(toutesCoordonnees) #[point gauche de la barre, point droit de la barre, coord hanche droite, coord hanche gauche, coord genou droite, coord genou gauche]
    descendreBarreValue = descendreBarre(toutesCoordonnees) #[point gauche de la barre, point droit de la barre]
    ramenerBarreControlerValue = ramenerBarreControler(toutesCoordonnees) #[point gauche de la barre, point droit de la barre, coord poignet droite, coord poignet gauche]
    bougerPiedsValue = bougerPieds(toutesCoordonnees[0]) #[coord talon droit, coord talon gauche]
    return [verouilletGenouxValue,epaulesArriereValue,barreVersBasValue,aiderCuisseValue,descendreBarreValue,ramenerBarreControlerValue,bougerPiedsValue]