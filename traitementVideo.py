import cv2
import numpy as np
from enumIndice import *
from utils import *
from squatDataRetrieve import *
from regleSouleveDeTerre import souleverDeTerreValue
from regleDeveloppeCouche import developpeCoucheCoord
from squatRules import SquatAnalyzer, analyze_sequence

# Pour le mode OFFLINE du squat
_squat_keypoints_collection = []  # Collecte des keypoints pendant la vidéo
_squat_analysis_result = None  # Résultat d'analyse à la fin

def rien(toutesCoordonnees: list,image: np.ndarray, winName : str) -> None:
    """Cette fonction permet de ne pas avoir de traitement des donnes pour tester si juste la recherche d'articulations fonctionne
    Args:
        toutesCoordonnees (list): toutes les coordonnees depuis le début de l'enregistrement
        image (np.ndarray): l'image à modifier"""
    cv2.imshow(winName, image)

def traitementVideoAngle3coordonnees(toutesCoordonnees : list,image : np.ndarray, winName : str) -> None:
    """calcul l'angle entre entre l'épaule droite le coude droit et le poignée droit et mets les points verts sur les articulations
    Args:
        toutesCoordonnees (list): toutes les coordonnees depuis le début de l'enregistrement
        image (np.ndarray): l'image à modifier"""
    coordonneesActuelle = toutesCoordonnees[-1]
    for point in coordonneesActuelle:
        cv2.circle(image, point, 5, (0, 255, 0), -1)
    if len(coordonneesActuelle) == 33 :
        angle = calculerAngle(coordonneesActuelle,IndiceMediaPipe.EPAULE_DROITE.value,IndiceMediaPipe.COUDE_DROIT.value,IndiceMediaPipe.POIGNET_DROIT.value)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Angle: {angle:.2f}°"
        cv2.putText(image, text, (image.shape[1] - 250, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(winName, image)

def Art(personnes: list, image: np.ndarray) -> None :
    """Mets les points d'articulation de toutes les personnes captées sur l'image
    Args:
        personnes (list): Liste des personnes captées avec leur coordonnées
        image (np.ndarray): L'image à modifier"""
    for personne in personnes :
        for point in personne :
            cv2.circle(image, point, 5, (0, 255, 0), -1)

def Box(boxes: list, image: np.ndarray) -> None :
    """Mets les rectangles de toutes les boxes captées sur l'image
    Args:
        boxes (list): Liste des boxes captées avec leur sommets
        image (np.ndarray): L'image à modifier"""
    for box in boxes :
        xMin, yMin, xMax, yMax = box
        cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (0, 255, 0), 2)


def affArt(toutesCoordonnees: list, image: np.ndarray, winName : str) -> None:
    """Affiche les articulations de toutes les personnes captées sur l'image
    Args:
        toutesCoordonnees (list): Regroupe toutes les coordonnées depuis le début de la vidéo
        image (np.ndarray): L'image à modifier"""
    Art(toutesCoordonnees[-1][0],image)
    cv2.imshow(winName, image)

def affBox(toutesCoordonnees: list, image: np.ndarray, winName : str) -> None:
    """Affiche les boxes de toutes les personnes captées sur l'image
    Args:
        toutesCoordonnees (list): Regroupe toutes les coordonnées depuis le début de la vidéo
        image (np.ndarray): L'image à modifier"""
    Box(toutesCoordonnees[-1][1],image)
    cv2.imshow(winName, image)

def affArtBox(toutesCoordonnees: list, image: np.ndarray, winName : str) -> None:
    """Affiche les boxes et les articulations de toutes les personnes captées sur l'image
    Args:
        toutesCoordonnees (list): Regroupe toutes les coordonnées depuis le début de la vidéo
        image (np.ndarray): L'image à modifier"""
    Art(toutesCoordonnees[-1][0],image)
    Box(toutesCoordonnees[-1][1],image)
    cv2.imshow(winName, image)

def affAngleGenouxSouleverTerre(toutesCoordonnees: list, image: np.ndarray, anglesGenoux : list) -> None:
    """Affiche les information sur l'angle du genoux pour le soulever de Terre
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        anglesGenoux (list): les données de l'angle du genoux"""
    if len(anglesGenoux) != 0 :
        toutesCoordonnees = toutesCoordonnees[0][0]
        cv2.circle(image, toutesCoordonnees[IndiceYolo.HANCHE_DROITE.value], 5, (0, 255, 0), -1)
        cv2.circle(image, toutesCoordonnees[IndiceYolo.GENOU_DROIT.value], 5, (0, 255, 0), -1)
        cv2.circle(image, toutesCoordonnees[IndiceYolo.TALON_DROIT.value], 5, (0, 255, 0), -1)
        cv2.line(image, toutesCoordonnees[IndiceYolo.HANCHE_DROITE.value], toutesCoordonnees[IndiceYolo.GENOU_DROIT.value], (0, 255, 0),thickness=2)
        cv2.line(image, toutesCoordonnees[IndiceYolo.GENOU_DROIT.value], toutesCoordonnees[IndiceYolo.TALON_DROIT.value], (0, 255, 0),thickness=2)
        cv2.putText(image, f"Angle du genoux droit : {anglesGenoux[0]:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(image, toutesCoordonnees[IndiceYolo.HANCHE_GAUCHE.value], 5, (0, 255, 0), -1)
        cv2.circle(image, toutesCoordonnees[IndiceYolo.GENOU_GAUCHE.value], 5, (0, 255, 0), -1)
        cv2.circle(image, toutesCoordonnees[IndiceYolo.TALON_GAUCHE.value], 5, (0, 255, 0), -1)
        cv2.line(image, toutesCoordonnees[IndiceYolo.HANCHE_GAUCHE.value], toutesCoordonnees[IndiceYolo.GENOU_GAUCHE.value], (0, 255, 0),thickness=2)
        cv2.line(image, toutesCoordonnees[IndiceYolo.GENOU_GAUCHE.value], toutesCoordonnees[IndiceYolo.TALON_GAUCHE.value], (0, 255, 0),thickness=2)
        cv2.putText(image, f"Angle du genoux gauche : {anglesGenoux[1]:.2f}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

def affEpauleArriere(toutesCoordonnees: list, image: np.ndarray, epauleArriere : list) -> None :
    """Affiche les information sur les épaules en arrière ( pour l'intant vide car les informations déjà afficher pour une autre vérification dans le mouvement)
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        epauleArriere (list): les données pour vérifier les épaules"""
    pass

def affBarreVersBas(toutesCoordonnees: list, image: np.ndarray, pointBarreEpaule : list) -> None:
    """Affiche les information de si la barre est descendu pendnat la monté
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        pointBarreEpaule (list): les données pour vérifier si la barre est descendu pendant la monté"""
    if len(pointBarreEpaule) != 0 :
        cv2.circle(image, pointBarreEpaule[2], 5, (0, 255, 0), -1)
        cv2.circle(image, pointBarreEpaule[3], 5, (0, 255, 0), -1)

def affAiderCuisse(toutesCoordonnees: list, image: np.ndarray, aiderCuisse : list) -> None:
    """Affiche les information pour savoir si il y a eu un aide des cuisses (pareil que affEpauleArriere)
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        aiderCuisse (list): Les données pour vérifier si il y a eu aide des cuisse"""
    pass

def affRedescendreBarre(toutesCoordonnees: list, image: np.ndarray, pointBarre : list) -> None:
    """Affiche les information pour savoir si la barre descend bien à partir de l'ordre donné
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        pointBarre (list): Les données pour vérifier si la barre descend bien à partir de l'ordre donné"""
    if len(pointBarre) != 0 :
        cv2.circle(image, pointBarre[0], 5, (0, 255, 0), -1)
        cv2.circle(image, pointBarre[1], 5, (0, 255, 0), -1)
        cv2.line(image, pointBarre[0], pointBarre[1], (0, 255, 0),thickness=2)

def affRamenerBarreControler(toutesCoordonnees: list, image: np.ndarray, pointBarrePoignet : list) -> None:
    """Affiche les information pour savoir si la descente de la barre est controlé
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        pointBarrePoignet (list): Les données pour vérifier si la descente de la barre est controlé"""
    if len(pointBarrePoignet) != 0 :
        cv2.circle(image, pointBarrePoignet[2], 5, (0, 255, 0), -1)
        cv2.circle(image, pointBarrePoignet[3], 5, (0, 255, 0), -1)

def affBougerPieds(toutesCoordonnees: list, image: np.ndarray, bougerPieds : list) -> None:
    """Affiche les information pour savoir si les pieds ont bougé pendant le mouvement
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        bougerPieds (list): Les données pour vérifier si les pieds ont bougé pendant le mouvement"""
    if len(bougerPieds) != 0:
        cv2.circle(image, bougerPieds[0], 5, (0, 255, 0), -1)
        cv2.circle(image, bougerPieds[1], 5, (0, 255, 0), -1)

def affInfoSouleveTerre(toutesCoordonnees: list, image: np.ndarray, winName : str) -> None:
    """Affiche toute les informations pour savoir si le mouvement est réussi
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        winName (str): nom de la fenêtre où afficher les informations"""
    toutesCoordonnees = toutesCoordonnees[-1]
    value = souleverDeTerreValue(toutesCoordonnees)
    affAngleGenouxSouleverTerre(toutesCoordonnees,image,value[0])
    affEpauleArriere(toutesCoordonnees,image,value[1])
    affBarreVersBas(toutesCoordonnees,image,value[2])
    affAiderCuisse(toutesCoordonnees,image,value[3])
    affRedescendreBarre(toutesCoordonnees,image,value[4])
    affRamenerBarreControler(toutesCoordonnees,image,value[5])
    affBougerPieds(toutesCoordonnees,image,value[6])
    cv2.imshow(winName, image)
    
def affOrientationTete(toutesCoordonnees: list, image: np.ndarray, angleTete: list) -> None:
    """Affiche les information sur la tête
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        angleTete (list): Les coordonées de la tête"""
    if isinstance(angleTete, (int, float)) and angleTete != 0:  # Vérifie si angleTete est un nombre non nul
        cv2.putText(image, f"Orientation Tete: {angleTete:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def affContactBanc(toutesCoordonnees: list, image: np.ndarray, contact: list) -> None:
    """Affiche les information sur le contact avec le banc
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        angleTete (list): Les coordonées sur le contact avec le banc"""
    if isinstance(contact, list) and len(contact) > 0 and all(isinstance(pt, tuple) and len(pt) == 2 for pt in contact):
        for pt in contact:
            cv2.circle(image, pt, 5, (0, 0, 255), -1)  # Rouge
        cv2.putText(image, "Contact Banc", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

def affPiedPlat(toutesCoordonnees: list, image: np.ndarray, pieds: list) -> None:
    """Affiche les information sur le contact avec le banc
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        angleTete (list): Les coordonées sur le contact avec le banc"""
    if isinstance(pieds, list) and len(pieds) == 2 and all(isinstance(pt, tuple) and len(pt) == 2 for pt in pieds):
        cv2.circle(image, tuple(pieds[0]), 5, (0, 255, 0), -1)
        cv2.circle(image, tuple(pieds[1]), 5, (0, 255, 0), -1)

def affDescenteBarre(toutesCoordonnees: list, image: np.ndarray, descente: list) -> None:
    """Affiche les information sur la descente de la barre
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        angleTete (list): Les coordonées sur la descente de la barre"""
    if isinstance(descente, list) and len(descente) == 8:
        barre, epaule_gauche, epaule_droite, coude_gauche, coude_droite, hanche_gauche, hanche_droite, poitrine_abdo = descente
        points = [barre[0], barre[1], epaule_gauche, epaule_droite, coude_gauche, coude_droite, hanche_gauche, hanche_droite, poitrine_abdo]
        
        if all(isinstance(pt, tuple) and len(pt) == 2 for pt in points):
            couleurs = [(0, 255, 0)] * len(points) 
            for pt, couleur in zip(points, couleurs):
                x, y = int(pt[0]), int(pt[1]) 
                cv2.circle(image, (x, y), 5, couleur, -1)

            cv2.line(image, barre[0], barre[1], (0, 255, 255), 2)
            cv2.line(image, epaule_gauche, epaule_droite, (255, 165, 0), 2) 
            cv2.line(image, hanche_gauche, hanche_droite, (255, 0, 255), 2)
            cv2.putText(image, "Descente Barre", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

def affEspacementMain(toutesCoordonnees: list, image: np.ndarray, angleTete : list) -> None:
    """Affiche les information sur l'espacement des mains (pareil que affEpauleArriere)
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        angleTete (list): Les coordonées sur l'espacement des mains"""
    pass

def affDeveloppeCouche(toutesCoordonnees: list, image: np.ndarray, winName : str) -> None:
    """Affiche les information pour le mouvement developpe couche
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
        winName (str): nom de la fenêtre où afficher les informations"""
    toutesCoordonnees = toutesCoordonnees[-1]
    res = developpeCoucheCoord(toutesCoordonnees)
    affOrientationTete(toutesCoordonnees, image, res[0])
    affContactBanc(toutesCoordonnees, image, res[1])
    affPiedPlat(toutesCoordonnees, image, res[2])
    affEspacementMain(toutesCoordonnees, image, res[4])
    affDescenteBarre(toutesCoordonnees, image, res[5])
    cv2.imshow(winName, image)


def draw_skeleton(image: np.ndarray, keypoints: list) -> None:
    """Dessine un squelette complet avec articulations et connexions YOLO-Pose avec numéros
    Args:
        image (np.ndarray): L'image à modifier
        keypoints (list): Liste des points clés du squelette"""
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    
    # Vérifier que nous avons assez de keypoints
    if len(keypoints) < 17:
        return
    
    # Dessiner les articulations (points verts) avec numéros
    for idx, (x, y) in enumerate(keypoints):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
            # Ajouter le numéro du point
            cv2.putText(image, str(idx), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Dessiner les connexions (lignes vertes)
    for a, b in skeleton:
        if a < len(keypoints) and b < len(keypoints):
            xa, ya = keypoints[a]
            xb, yb = keypoints[b]
            if isinstance(xa, (int, float)) and isinstance(ya, (int, float)) and isinstance(xb, (int, float)) and isinstance(yb, (int, float)):
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)


def affichageOrienteSquat(toutesCoordonnees : list, image : np.ndarray, winName : str)-> None :
    """
    MODE OFFLINE: Collecte les keypoints durant la vidéo (pas d'analyse en temps réel)
    L'analyse se fera à la fin via analyze_sequence()
    
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier
    """
    global _squat_keypoints_collection
    
    listeArt, [anglesG, angleD] = traitementSquat(toutesCoordonnees)
    
    # Récupérer les keypoints complets (tous les 17 points YOLO)
    articulations = toutesCoordonnees[-1][0]
    if articulations:
        # Utiliser le squelette complet du premier athlète détecté
        full_keypoints = articulations[0]
        draw_skeleton(image, full_keypoints)
        
        # COLLECTER les keypoints (pas d'analyse encore)
        _squat_keypoints_collection.append(full_keypoints)
        
        # Afficher les informations de base (angle des genoux)
        y_offset = 30
        text_angle = f"Angle: G={anglesG:.1f}° D={angleD:.1f}°"
        cv2.putText(image, text_angle, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # Afficher le nombre de frames collectés
        text_frames = f"Frames collectés: {len(_squat_keypoints_collection)}"
        cv2.putText(image, text_frames, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    
    Box(toutesCoordonnees[-1][1], image)
    cv2.imshow(winName, image)


def analyze_squat_offline(fps=30):
    """
    Analyse complète du squat à la fin de la vidéo (OFFLINE).
    Utilise la classe SquatAnalyzerOffline pour une meilleure détection.
    
    Returns:
        dict avec verdict, défauts, etc.
    """
    global _squat_keypoints_collection, _squat_analysis_result
    
    if len(_squat_keypoints_collection) < 10:
        _squat_analysis_result = {
            "verdict": "INVALIDE",
            "defaut": None,
            "message": "Trop peu de frames collectés",
            "details": {}
        }
        return _squat_analysis_result
    
    # Analyse OFFLINE robuste
    _squat_analysis_result = analyze_sequence(_squat_keypoints_collection, fps=fps, view_id="center")
    return _squat_analysis_result


def display_squat_analysis(image : np.ndarray, analysis_result : dict, x_offset=10, y_offset=30):
    """
    Affiche l'analyse du squat sur l'image (à côté, pas sur le squat).
    
    Args:
        image: image où afficher
        analysis_result: résultat de analyze_sequence()
        x_offset, y_offset: position du texte
    """
    if analysis_result is None:
        return
    
    # Couleur basée sur le verdict
    color = (0, 0, 255) if analysis_result["verdict"] == "REFUSE" else (0, 255, 0)
    
    # Titre
    text_verdict = f"VERDICT: {analysis_result['verdict']}"
    cv2.putText(image, text_verdict, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    y_offset += 40
    
    # Défaut
    if analysis_result["defaut"]:
        text_defaut = f"Défaut: {analysis_result['defaut']}"
        cv2.putText(image, text_defaut, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        y_offset += 35
    
    # Message
    msg = analysis_result.get("message", "")
    if msg:
        # Découper le message s'il est trop long
        lines = [msg[i:i+50] for i in range(0, len(msg), 50)]
        for line in lines:
            cv2.putText(image, line, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 25
    
    # Détails
    details = analysis_result.get("details", {})
    if details:
        y_offset += 10
        text_bottom = f"Bottom frame: {analysis_result.get('bottom_frame', -1)}"
        cv2.putText(image, text_bottom, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        y_offset += 25
        
        if "depth_confidence" in details:
            text_depth = f"Depth confidence: {details['depth_confidence']:.2f}"
            cv2.putText(image, text_depth, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            y_offset += 25


def reset_squat_analysis():
    """Réinitialise la collection pour un nouvel essai"""
    global _squat_keypoints_collection, _squat_analysis_result
    _squat_keypoints_collection = []
    _squat_analysis_result = None


def tracerTrajectoire(image : np.ndarray, trajectoire : list):
    """Trace la trajectoire d'un objet en mouvement sur l'image.
    Args:
        image (numpy.ndarray): Image actuelle sur laquelle tracer la trajectoire.
        trajectoire (list): Liste des coordonnées (x, y) du centre de l'objet."""
    couleur=(0, 0, 255)
    for i in range(1, len(trajectoire)):
        if trajectoire[i - 1] is None or trajectoire[i] is None:
            continue
        cv2.line(image, trajectoire[i - 1], trajectoire[i], couleur, 2)
