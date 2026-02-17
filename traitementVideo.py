import cv2
import numpy as np
from enumIndice import *
from utils import *
from squatDataRetrieve import *
from regleSouleveDeTerre import souleverDeTerreValue
from regleDeveloppeCouche import developpeCoucheCoord
from squatRules import SquatAnalyzer

# Analyseur global pour le squat
_squat_analyzer = None
_squat_frame_counter = 0
# Variables pour garder le défaut affiché plusieurs frames
_last_defaut = None
_defaut_display_frames = 0
_DEFAUT_DISPLAY_DURATION = 120  # Garder le défaut affiché pendant 120 frames

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
    """Affiche les informations utile pour le squat avec détection des défauts
    Args:
        toutesCoordonnees (list): les coordonnées depuis le début du mouvement
        image (np.ndarray): l'image actuelle à modifier"""
    global _squat_analyzer, _squat_frame_counter, _last_defaut, _defaut_display_frames
    
    # Initialiser l'analyseur au premier frame
    if _squat_analyzer is None:
        _squat_analyzer = SquatAnalyzer()
        _squat_frame_counter = 0
    
    listeArt, [anglesG, angleD] = traitementSquat(toutesCoordonnees)
    
    # Récupérer les keypoints complets (tous les 17 points YOLO)
    articulations = toutesCoordonnees[-1][0]
    if articulations:
        # Utiliser le squelette complet du premier athlète détecté
        full_keypoints = articulations[0]
        draw_skeleton(image, full_keypoints)
        
        # Analyser le frame pour détecter les défauts
        analysis_result = _squat_analyzer.analyze_frame(full_keypoints, _squat_frame_counter)
        
        # Afficher les informations d'analyse
        y_offset = 30
        
        # Afficher l'angle des genoux
        text_angle = f"Angle: G={anglesG:.1f}° D={angleD:.1f}°"
        cv2.putText(image, text_angle, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # Afficher la phase du mouvement
        text_phase = f"Phase: {analysis_result['phase']}"
        cv2.putText(image, text_phase, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        y_offset += 30
        
        # Si défaut détecté, sauvegarder et afficher
        if analysis_result["verdict"] == "REFUSE":
            _last_defaut = analysis_result
            _defaut_display_frames = _DEFAUT_DISPLAY_DURATION
        
        # Afficher le défaut s'il y en a un en mémoire
        if _last_defaut is not None and _defaut_display_frames > 0:
            color = (0, 0, 255)  # Rouge
            text_verdict = f"DEFAUT: {_last_defaut['defaut']}"
            cv2.putText(image, text_verdict, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            y_offset += 35
            
            text_reason = _last_defaut["message"]
            cv2.putText(image, text_reason, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30
            
            text_frame = f"Frame du defaut: {_last_defaut['instant_defaut']}"
            cv2.putText(image, text_frame, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
            
            # Afficher le temps restant pour le défaut
            seconds_left = _defaut_display_frames / 30  # Supposant 30 FPS
            text_timer = f"Defaut visible pendant {seconds_left:.1f}s"
            cv2.putText(image, text_timer, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
            
            _defaut_display_frames -= 1
        else:
            if _defaut_display_frames <= 0:
                _last_defaut = None
            text_verdict = "Statut: OK"
            cv2.putText(image, text_verdict, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        _squat_frame_counter += 1
    
    Box(toutesCoordonnees[-1][1], image)
    cv2.imshow(winName, image)


def get_squat_verdict():
    """Retourne le verdict final du squat analysé"""
    global _squat_analyzer
    if _squat_analyzer is None:
        return {
            "verdict": "INCONNU",
            "defaut": None,
            "message": "Aucun squat analysé"
        }
    return _squat_analyzer.get_result()


def reset_squat_analyzer():
    """Réinitialise l'analyseur pour une nouvelle analyse"""
    global _squat_analyzer, _squat_frame_counter
    _squat_analyzer = None
    _squat_frame_counter = 0


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
