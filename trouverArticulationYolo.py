import mediapipe as mp
import cv2
import numpy as np
from enumIndice import *
from ultralytics import YOLO
import math
from traitementVideo import choixDevice

mouseX, mouseY = -1,-1

def mouseCallBack(event : int, x : int, y : int, flags : int, param):
    """Callback permettant de récupérer les coordonnées de la souris lors d'un mouvement
    Args:
        event (int): Type d'événement déclenché par la souris
        x (int): Coordonnée X de la souris
        y (int): Coordonnée Y de la souris
        flags (int): Indicateurs supplémentaires pour les événements de la souris
        param (any): Paramètre optionnel pouvant être utilisé pour transmettre des données supplémentaires
    Returns:
        None: Cette fonction ne retourne rien, elle met à jour les coordonnées globales de la souris"""
    global mouseX, mouseY
    if event == cv2.EVENT_MOUSEMOVE:
        mouseX, mouseY = x, y

def detecterYolo(image: np.ndarray, modelYolo: YOLO, conf: float = 0.5, maxDet: int = 10) -> list:
    """Détecte les objets dans une image avec YOLO.
    Args:
        image (np.ndarray): L'image à analyser.
        modelYolo (YOLO): Le modèle YOLO pour la détection des objets.
        conf (float, optional): Seuil de confiance minimal. Par défaut à 0.5.
        maxDet (int, optional): Nombre maximal de détections. Par défaut à None.
    Returns:
        list: Résultats des détections YOLO"""
    model = modelYolo[0]
    device = choixDevice()
    return model.track(source=image,
                       persist=True, 
                       conf=conf,
                       verbose=False, 
                       stream=True, 
                       save=False, 
                       max_det=maxDet, 
                       tracker="bytetrack.yaml", 
                       device=device,
                       agnostic_nms = True)

def extraireArticulation(results : any) -> list:
    """Extrait les points clés des résultats YOLO.
    Args:
        results: Résultats YOLO
    Returns:
        list: Liste des coordonnées des points clés"""
    keypoints_list = []
    for result in results:
        for keypoints in result.keypoints.xy:
            keypoints_list.append([(int(p[0]), int(p[1])) for p in keypoints])
    return keypoints_list

def extraireBoxes(results : any) -> list:
    """Extrait les boîtes englobantes des résultats YOLO.
    Args:
        results: Résultats YOLO
    Returns:
        list: Liste des boîtes englobantes [(x_min, y_min, x_max, y_max), ...]"""
    boxes = []
    for result in results:
        for box in result.boxes.xyxy:
            xMin, yMin, xMax, yMax = map(int, box.tolist())
            boxes.append((xMin, yMin, xMax, yMax))
    return boxes


def extraireArtBox(results : any) -> list:
    """Extrait les point clés et les boîtes englobantes des résultats YOLO
    Args:
        results (any): Résultats du modèle Yolo
    Returns:
        list: Tuple contenant la liste des keypoints et des boîtes englobantes"""
    keypoints_list, boxes = [], []

    for result in results:
        for keypoints in result.keypoints.xy:
            keypoints_list.append([(int(p[0]), int(p[1])) for p in keypoints])
        for box in result.boxes.xyxy:
            xMin, yMin, xMax, yMax = map(int, box.tolist())
            boxes.append((xMin, yMin, xMax, yMax))

    return keypoints_list, boxes

def meilleurBoxeCoord(boxes: list, xRef: int, yRef: int) -> list:
    """Trouve la boîte englobante la plus proche d'un point de référence.
    Args:
        boxes (list): Liste des boîtes englobantes.
        xRef (int): Coordonnée X de référence.
        yRef (int): Coordonnée Y de référence.
    Returns:
        list: La boîte la plus proche [(x_min, y_min, x_max, y_max)] ou [] si aucune boîte."""
    best_box = None
    min_offset = float('inf')
    
    for xMin, yMin, xMax, yMax in boxes:
        boxCentreX, boxCentreY = (xMin + xMax) // 2, (yMin + yMax) // 2
        offset = abs(xRef - boxCentreX) + abs(yRef - boxCentreY)
        if offset < min_offset:
            min_offset = offset
            best_box = (xMin, yMin, xMax, yMax)
    
    return [best_box] if best_box else []

def meilleurBoxeArea(results :any) -> list:
    """Retourne la boîte englobante avec la plus grande aire.
    Args:
        results: Résultats YOLO contenant les boîtes détectées.
    Returns:
        list: La boîte englobante ayant la plus grande aire [(x_min, y_min, x_max, y_max)] ou [] si aucune boîte."""
    max_box = None
    max_area = 0
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                area = (x_max - x_min) * (y_max - y_min)
                if area > max_area:
                    max_area = area
                    max_box = (x_min, y_min, x_max, y_max)
    
    return [max_box] if max_box else []

def yolo1Art(image: np.ndarray, modelMP: mp.solutions.pose.Pose, modelYolo: YOLO) -> list:
    """Permet d'obtenir 17 points d'articulations via le modèle YoloVX.
    Args:
        image (np.ndarray): L'image à analyser.
        modelYolo (YOLO): Le modèle YOLO pour la détection.
    Returns:
        list: Liste des coordonnées des points d'articulations."""
    results = detecterYolo(image, modelYolo, conf=0.3)
    return extraireArticulation(results)[0] if results else []

def yoloArt(image: np.ndarray, modelMP: mp.solutions.pose.Pose, modelYolo: YOLO) -> list:
    """Permet d'obtenir les points d'articulations de jusqu'à cinq personnes via le modèle YoloVX.
    Args:
        image (np.ndarray): L'image à analyser.
        modelYolo (YOLO): Le modèle YOLO pour la détection.
    Returns:
        list: Liste contenant les coordonnées des points d'articulations de chaque personne détectée."""
    results = detecterYolo(image, modelYolo, conf=0.25)
    return extraireArticulation(results)

def yoloAllBox(image: np.ndarray, modelMP: mp.solutions.pose.Pose, modelYolo: YOLO) -> list:
    """Détecte toutes les boîtes englobantes des personnes présentes dans l'image avec YOLO.
    Args:
        image (np.ndarray): L'image à analyser.
        modelYolo (YOLO): Le modèle YOLO pour la détection.
    Returns:
        list: Liste des boîtes englobantes."""
    results = detecterYolo(image, modelYolo)
    return extraireBoxes(results)

def yoloCentreBox(image: np.ndarray, modelMP: mp.solutions.pose.Pose, modelYolo: YOLO) -> list:
    """Détecte la boîte englobante la plus centrée dans l'image avec YOLO.
    Args:
        image (np.ndarray): L'image à analyser.
        modelYolo (YOLO): Le modèle YOLO pour la détection.
    Returns:
        list: La boîte la plus centrée [(x_min, y_min, x_max, y_max)]."""
    xRef, yRef = image.shape[1] // 2, image.shape[0] // 2
    return meilleurBoxeCoord(yoloAllBox(image, modelYolo), xRef, yRef)

def yoloSourisBox(image: np.ndarray, modelMP: mp.solutions.pose.Pose, modelYolo: YOLO) -> list:
    """Détecte la boîte englobante la plus proche de la position de la souris avec YOLO.
    Args:
        image (np.ndarray): L'image à analyser.
        modelYolo (YOLO): Le modèle YOLO pour la détection.
    Returns:
        list: La boîte la plus proche [(x_min, y_min, x_max, y_max)]."""
    xRef, yRef = (mouseX, mouseY) if (mouseX, mouseY) != (-1, -1) else (image.shape[1] // 2, image.shape[0] // 2)
    return meilleurBoxeCoord(yoloAllBox(image,modelMP, modelYolo), xRef, yRef)

def yoloBarreBox(image: np.ndarray, modelMP: mp.solutions.pose.Pose, modelYolo: YOLO) -> list:
    """Détecte la boîte englobante de la barre en métal avec YOLO.
    Args:
        image (np.ndarray): L'image à analyser.
        modelMP (mp.solutions.pose.Pose): Modèle Mediapipe (non utilisé ici).
        modelYolo (YOLO): Le modèle YOLO utilisé pour la détection.
    Returns:
        list: La boîte englobante de la barre [(x_min, y_min, x_max, y_max)] ou []."""
    results = detecterYolo(image, modelYolo, conf=0.15, maxDet=5)
    return meilleurBoxeArea(results)

def calculer_distance(point1, point2) -> float:
    """Calcule la distance euclidienne entre deux points.
    Args:
        point1 (tuple): Coordonnées du premier point (x, y).
        point2 (tuple): Coordonnées du second point (x, y).
    Returns:
        float: Distance euclidienne entre les deux points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distancePersonneBarre(coordonnees : list, box : list, image) -> list:
    """Renvoie une liste des distances entre chaque personne et la barre de musculation.
    Args:
        coordonnees (list): Liste des coordonnées des personnes détectées.
        box (list): Coordonnées de la boîte englobante de la barre.
    Returns:
        list: Distance entre chaque personne et la barre."""
    x_min, y_min, x_max, y_max = box[0]
    x_centre, y_centre = (x_min + x_max) // 2, (y_min + y_max) // 2
    distances = list()

    for personne in coordonnees:
        poignet_droit, poignet_gauche, nez = personne[IndiceYolo.POIGNET_DROIT.value], personne[IndiceYolo.POIGNET_GAUCHE.value], personne[IndiceYolo.NEZ.value]
        point_ref = None
        div = 3

        if all(i == 0 for i in poignet_droit):
            div -= 1
        if all(i == 0 for i in poignet_gauche):
            div -= 1
        if all(i == 0 for i in nez):
            div -= 1
        if div == 0:
            break
        else:
            x_nez, y_nez = nez
            x_poignet_droit, y_poignet_droit = poignet_droit
            x_poignet_gauche, y_poignet_gauche = poignet_gauche
            point_ref = ((x_nez + x_poignet_droit + x_poignet_gauche) // div), ((y_nez + y_poignet_droit + y_poignet_gauche) // div)
            cv2.circle(image, point_ref, 5, (255, 255, 255), -1)
            distances.append(calculer_distance(point_ref, (x_centre, y_centre)))

    return distances


def personneProcheBarre(coordonnees: list, box: list, image: np.ndarray) -> int:
    """Détermine l'indice de la personne la plus proche de la barre détectée.
    Args:
        coordonnees (list): Liste des coordonnées des personnes détectées.
        box (list): Coordonnées de la boîte englobante de la barre.
        image (np.ndarray): L'image à analyser.
    Returns:
        int: Indice de la personne la plus proche de la barre ou -1 si aucune personne détectée."""
    if not box or not coordonnees:
        return -1
    
    distances = distancePersonneBarre(coordonnees, box, image)
    return distances.index(min(distances)) if distances else -1

def yoloProcheBarreArtBox(image: np.ndarray, modelMP: mp.solutions.pose.Pose, modelYolo: YOLO) -> tuple[list, list]:
    """Retourne les articulations de la personne la plus proche de la barre ainsi que la boîte de la barre.
    Args:
        image (np.ndarray): L'image à analyser.
        modelMP (mp.solutions.pose.Pose): Modèle Mediapipe (non utilisé ici).
        modelYolo (YOLO): Le modèle YOLO utilisé pour la détection.
    Returns:
        tuple[list, list]: Liste des articulations de la personne la plus proche et la boîte englobante de la barre."""
    box = yoloBarreBox(image, modelMP, [modelYolo[1]])
    coord = yoloArt(image, modelMP, [modelYolo[0]])
    idPers = personneProcheBarre(coord, box, image)
    return ([coord[idPers]], box) if idPers != -1 else ([], [])

def yoloVOT(image : np.ndarray, modelMP: mp.solutions.pose.Pose, modelYolo : list) -> tuple[list, list]:
    """Renvoie les coordonnées de la boite contenant la personne détecté par le modele
    Args:
        image (np.ndarray): L'image à analyser
        modelMP (mp.solutions.pose.Pose): Modèle Mediapipe (non utilisé ici -> pour la généricité des fonctions).
        modelYolo (YOLO): Le modèle YOLO utilisé
    Returns:
        list: Liste des coordonnées de la personne
        list: Liste des coordonnées de la boite englobant la personne"""
    device = choixDevice()
    resultsByteTrack = modelYolo[0].track(image, 
                                        persist=True, 
                                        verbose = False, 
                                        classes=0,
                                        stream=True, 
                                        conf=0.3, 
                                        tracker="bytetrack.yaml", 
                                        device=device, 
                                        agnostic_nms = True)
    art, box = extraireArtBox(resultsByteTrack)
    return (art, box)

def calculerVitessePixels(box : list, fps : float) -> float:
    """Calcule la vitesse moyenne en pixels par seconde d'un objet
    Args:
        coordonnees (list): Liste des positions l'objet à chaque frame.
        fps (float): Nombre d'images par seconde de la vidéo.
    Returns:
        float: Vitesse moyenne en pixels par seconde."""
    if len(box) < 2 or fps <= 0:
        return 0
    
    distance_totale_pixels = np.sum(np.sqrt(np.diff([coord[0] for coord in box])**2 + np.diff([coord[1] for coord in box])**2))
    duree_totale = len(box) / fps
    
    return distance_totale_pixels / duree_totale if duree_totale > 0 else 0


def calculerDistancePixels(box : list) -> float:
    """
    Calcule la distance totale parcourue en pixels par un objet
    Args:
        box (list): Liste des positions de l'objet au fil des frames.
    Returns:
        float: Distance totale parcourue en pixels."""
    distance_totale = 0
    for i in range(1, len(box)):
        x1, y1 = box[i - 1]
        x2, y2 = box[i]
        distance_totale += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return distance_totale