import cv2
import numpy as np
from enumIndice import *
from traitementVideo import *
from trouverArticulationYolo import *
import time
from onnxOptimiser import *
from exportationDonneeCSV import exportationCSV, incrementerIndice
import mediapipe as mp

# ========== CONSTANTES DE TRACKING ==========
MAX_DISTANCE_BOX = 500  # Distance max pour matcher (augmenté de 300 pour squat)
HISTORY_SIZE = 5  # Nombre de frames précédentes à garder
SCALE_FACTOR = 0.8  # Multiplicateur pour agrandir la zone de recherche progressivement


# =========================
# Helpers sélection ATHLETE
# =========================
_selected_index = None
_clicked_point = None

def _draw_numbered_boxes(img, boxes, color=(0, 255, 0)):
    """Dessine les boxes + numéros."""
    for i, (xMin, yMin, xMax, yMax) in enumerate(boxes):
        cv2.rectangle(img, (xMin, yMin), (xMax, yMax), color, 2)
        cv2.putText(img, str(i), (xMin + 5, yMin + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def _click_select_box(event, x, y, flags, param):
    """Callback souris: clique dans une box => sélectionne son index."""
    global _selected_index, _clicked_point
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    boxes = param
    _clicked_point = (x, y)
    for i, (xMin, yMin, xMax, yMax) in enumerate(boxes):
        if xMin <= x <= xMax and yMin <= y <= yMax:
            _selected_index = i
            break

def _box_center(box):
    """...existing code..."""
    xMin, yMin, xMax, yMax = box
    return ((xMin + xMax) / 2.0, (yMin + yMax) / 2.0)

def _box_area(box):
    xMin, yMin, xMax, yMax = box
    return (xMax - xMin) * (yMax - yMin)

def _find_closest_box(boxes, target_box, max_distance=200):
    """
    Trouve la box la plus proche du target_box.
    max_distance: distance max en pixels pour considérer c'est le même athlète
    Retourne (index, distance) ou (None, inf) si pas trouvé
    """
    if not boxes or target_box is None:
        return None, float("inf")
    
    tx, ty = _box_center(target_box)
    best_i = None
    best_d = float("inf")
    
    for i, b in enumerate(boxes):
        cx, cy = _box_center(b)
        d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if d < best_d and d <= max_distance:
            best_d = d
            best_i = i
    
    return best_i, best_d

def _find_closest_box_with_history(boxes, target_box, history_boxes, max_distance=None):
    """
    Trouve la box la plus proche du target_box avec historique de mouvement.
    - Prédit la position suivante basée sur l'historique
    - Augmente progressivement la distance de recherche si rien trouvé
    - FILTRE LES AUTRES PERSONNES pour éviter de sauter à une personne voisine
    
    Args:
        boxes: boxes actuelles
        target_box: dernière box connue de l'athlète
        history_boxes: liste des 5 dernières boxes
        max_distance: distance max initiale (utilise MAX_DISTANCE_BOX par défaut)
    
    Returns:
        (index, distance, predicted_center) ou (None, inf, None)
    """
    if not boxes or target_box is None:
        return None, float("inf"), None
    
    if max_distance is None:
        max_distance = MAX_DISTANCE_BOX
    
    tx, ty = _box_center(target_box)
    
    # ========== FILTRER LES AUTRES PERSONNES ==========
    # Si on a plusieurs boxes, ignorer celles qui sont trop éloignées
    # pour éviter de "sauter" à une autre personne
    if len(boxes) > 1:
        filtered_boxes_with_idx = []
        for i, b in enumerate(boxes):
            cx, cy = _box_center(b)
            d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            # Garder seulement les boxes dans une zone raisonnable
            if d <= max_distance:
                filtered_boxes_with_idx.append((i, b, d))
        
        # Si aucune box n'est assez proche, utiliser le seuil plus large
        if not filtered_boxes_with_idx:
            # Ignorer toutes les boxes sauf la plus proche
            closest_idx = min(range(len(boxes)), 
                            key=lambda i: (((_box_center(boxes[i])[0] - tx) ** 2 + 
                                          (_box_center(boxes[i])[1] - ty) ** 2) ** 0.5))
            filtered_boxes_with_idx = [(closest_idx, boxes[closest_idx], 
                                      (((_box_center(boxes[closest_idx])[0] - tx) ** 2 + 
                                       (_box_center(boxes[closest_idx])[1] - ty) ** 2) ** 0.5))]
    else:
        filtered_boxes_with_idx = [(0, boxes[0], 0)]
    
    # ========== PRÉDICTION SIMPLE ==========
    predicted_x, predicted_y = tx, ty
    
    if len(history_boxes) >= 2:
        prev_x, prev_y = _box_center(history_boxes[-2])
        curr_x, curr_y = _box_center(history_boxes[-1])
        
        motion_x = (curr_x - prev_x) * 0.5
        motion_y = (curr_y - prev_y) * 0.5
        
        predicted_x = curr_x + motion_x
        predicted_y = curr_y + motion_y
    
    # ========== RECHERCHE SUR BOXES FILTRÉES ==========
    best_i = None
    best_d = float("inf")
    
    for idx, b, _ in filtered_boxes_with_idx:
        cx, cy = _box_center(b)
        d = ((cx - predicted_x) ** 2 + (cy - predicted_y) ** 2) ** 0.5
        
        if d < best_d:
            best_d = d
            best_i = idx
    
    return best_i, best_d, (predicted_x, predicted_y)

def _extract_mediapipe_skeleton(image, mediapipe_model):
    """
    Utilise MediaPipe pour extraire le squelette si YOLO échoue.
    Retourne une liste contenant les keypoints MediaPipe.
    """
    if mediapipe_model is None:
        return None
    
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mediapipe_model.process(image_rgb)
        
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                keypoints.append((x, y))
            return [keypoints]  # Retourner dans le format attendu
        else:
            return None
    except Exception as e:
        print(f"[WARN] Erreur MediaPipe: {e}")
        return None

def _reorder_by_target(arts, boxes, target_box):
    """
    Réordonne arts/boxes pour mettre l'athlète en index 0.
    """
    if not boxes or not arts or target_box is None:
        return arts, boxes, target_box

    tx, ty = _box_center(target_box)
    best_i = 0
    best_d = float("inf")

    for i, b in enumerate(boxes):
        cx, cy = _box_center(b)
        d = (cx - tx) ** 2 + (cy - ty) ** 2
        if d < best_d:
            best_d = d
            best_i = i

    if best_i != 0:
        arts = [arts[best_i]] + [a for k, a in enumerate(arts) if k != best_i]
        boxes = [boxes[best_i]] + [b for k, b in enumerate(boxes) if k != best_i]

    target_box = boxes[0]
    return arts, boxes, target_box


def initialisationDataVideo(videoPath: str, id: int):
    if videoPath is None:
        videoPath = 0
        angle = 0
    elif videoPath.isdigit():
        videoPath = int(videoPath)
        angle = 0
    else:
        angle = 0
        if angle is None:
            angle = 0

    capture = cv2.VideoCapture(videoPath)
    # NE PAS forcer la taille pour fichiers vidéo (webcam ok)
    if isinstance(videoPath, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, LARGEUR)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HAUTEUR)
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    # Récupérer les dimensions natives de la vidéo
    W = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    winName = "Pose Detection" + str(id)
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winName, W, H)
    cv2.setWindowProperty(winName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(winName, mouseCallBack)
    return capture, angle, winName, W, H


def captureVideo(mediaPipe: bool = True,
                 yolo: list = [],
                 trouverArticulation=yoloVOT,
                 videoPath: str = None,
                 enregistrerVideoAvant: str = None,
                 traitementVideo=rien,
                 choixMateriel: bool = False,
                 enregistrementVideoApres: str = None,
                 enregistrementJson: str = None,
                 id: int = 0):

    global _selected_index, _clicked_point

    tempsTotalFrame, nbTotalFrame = 0, 0
    print(f"Lancement du logiciel avec mediaPipe : {mediaPipe}, yolo : {yolo}, et comme path pour la vidéo : {videoPath}")

    modelMP = None
    if mediaPipe:
        mp_pose = mp.solutions.pose
        modelMP = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full (plus robuste)
            smooth_landmarks=True,
            min_detection_confidence=0.3,  # Baisser pour plus de sensibilité
            min_tracking_confidence=0.3
        )
    
    modelYolo = yolo
    capture, angleVideo, winName, W, H = initialisationDataVideo(videoPath, id)
    if not capture.isOpened():
        print(f"[ERREUR Process {id}] Échec de l'ouverture de la capture pour: {videoPath}")
        return

    if enregistrerVideoAvant is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outAvant = cv2.VideoWriter(enregistrerVideoAvant, fourcc, 20.0, (W, H))
    if enregistrementVideoApres is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outApres = cv2.VideoWriter(enregistrementVideoApres, fourcc, 20.0, (W, H))

    touteCoordonnees = []

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
        print("Erreur: FPS non récupéré. Utilisation de 30.")

    trajectoire = []
    pov = ...

    mouvement = ...
    if traitementVideo == affDeveloppeCouche:
        mouvement = IndiceMouvement.BENCH.value
    elif traitementVideo == affichageOrienteSquat:
        mouvement = IndiceMouvement.SQUAT.value
    else:
        mouvement = IndiceMouvement.DEADLIFT.value

    # =============================
    # Etat sélection ATHLETE + TRACKING
    # =============================
    athlete_locked = False
    target_box = None
    history_boxes = []  # Garder les dernières N positions
    last_valid_keypoints = None
    frames_without_detection = 0
    MAX_FRAMES_WITHOUT_YOLO = 5  # Réduit de 10 à 5 (MediaPipe + prédiction)

    paused = False

    while capture.isOpened():
        retour, image = capture.read()
        if not retour:
            print("Une erreur avec la capture de l'image.")
            break

        debut = time.time()

        # Détection YOLO
        coordonnees = trouverArticulation(image, modelMP, modelYolo)
        if enregistrerVideoAvant is not None:
            outAvant.write(cv2.resize(image.copy(), (LARGEUR, HAUTEUR), interpolation=cv2.INTER_AREA))

        arts, boxes = coordonnees[0], coordonnees[1]

        # ============ LOGIQUE DE TRACKING AMÉLIORÉE ============
        if athlete_locked:
            if boxes and arts:
                frames_without_detection = 0
                
                # Chercher avec historique + prédiction + FILTRAGE DES AUTRES PERSONNES
                closest_idx, distance, predicted = _find_closest_box_with_history(
                    boxes, target_box, history_boxes, max_distance=MAX_DISTANCE_BOX
                )
                
                if closest_idx is not None:
                    # Réordonner
                    arts = [arts[closest_idx]] + [a for k, a in enumerate(arts) if k != closest_idx]
                    boxes = [boxes[closest_idx]] + [b for k, b in enumerate(boxes) if k != closest_idx]
                    target_box = boxes[0]
                    last_valid_keypoints = arts[0]
                    
                    # Mettre à jour l'historique
                    history_boxes.append(target_box)
                    if len(history_boxes) > HISTORY_SIZE:
                        history_boxes.pop(0)
                    
                    # Log seulement si distance anormale
                    if distance > 200:
                        print(f"[TRACK] Distance: {distance:.1f}px (en mouvement)")
                
                else:
                    # Pas trouvé => MediaPipe fallback
                    print(f"[WARN] Athlète perdu mais continuité maintenue. Fallback MediaPipe...")
                    mp_keypoints = _extract_mediapipe_skeleton(image, modelMP)
                    if mp_keypoints:
                        arts = mp_keypoints
                        boxes = []
                        last_valid_keypoints = arts[0]
                    else:
                        # Garder le dernier squelette connu
                        if last_valid_keypoints:
                            arts = [last_valid_keypoints]
                            boxes = []
        else:
            # Athlète non verrouillé => utiliser YOLO normalement
            if boxes and arts:
                arts = [arts[0]] + [a for k, a in enumerate(arts) if k != 0]
                boxes = [boxes[0]] + [b for k, b in enumerate(boxes) if k != 0]

        coordonnees = (arts, boxes)
        touteCoordonnees.append(coordonnees)

        # --------- Trajectoire ----------
        if len(touteCoordonnees[-1][1]) > 0:
            xMin, yMin, xMax, yMax = touteCoordonnees[-1][1][0]
            if pov == IndicePointDeVue.GAUCHE.value:
                boutBarre = (xMin, int((yMin + yMax) / 2))
            elif pov == IndicePointDeVue.DROITE.value:
                boutBarre = (xMax, int((yMin + yMax) / 2))
            else:
                boutBarre = (int(xMin + (xMax - xMin) / 2), yMax)
            trajectoire.append(boutBarre)

        # Traitement métier
        mouvementReussi = traitementVideo(touteCoordonnees, image, winName)

        # Afficher image à résolution native (pas de resize)
        image_display = image.copy()
        
        # Afficher status
        if athlete_locked:
            status_text = "LOCKED"
            status_color = (0, 255, 0)  # Vert
            
            if not boxes:
                status_text = "LOCKED (MediaPipe fallback)"
                status_color = (0, 165, 255)  # Orange
            elif len(history_boxes) > 1:
                status_text = f"LOCKED (tracking: {len(history_boxes)} frames)"
            
            cv2.putText(image_display, f"[{status_text}]", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow(winName, image_display)

        if enregistrementJson is not None:
            exportationCSV(touteCoordonnees, enregistrementJson, mouvement)

        fin = time.time()
        if enregistrementVideoApres is not None:
            outApres.write(image)

        # =============================
        # Gestion clavier + Fermeture
        # =============================
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or cv2.getWindowProperty(winName, cv2.WND_PROP_VISIBLE) < 1:
            break

        # ---- Appui sur V => PAUSE + Sélection ----
        if key == ord('v'):
            paused = True
            _selected_index = None
            _clicked_point = None

            # Pas de resize: utiliser l'image native
            sel_img = image.copy()
            
            # Pas besoin de conversion d'échelle (sx=sy=1.0)
            # Les boxes sont déjà en coordonnées natives
            _draw_numbered_boxes(sel_img, boxes)

            sel_win = f"Selection Athlete {id}"
            cv2.namedWindow(sel_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(sel_win, W, H)
            cv2.moveWindow(sel_win, 50 + W, 50)

            while paused:
                cv2.imshow(sel_win, sel_img)
                # Afficher les instructions au clavier
                img_with_text = sel_img.copy()
                cv2.putText(img_with_text, f"Cadres disponibles: 0-{len(boxes)-1}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img_with_text, "Tapez le numero du cadre (0-9) + ENTREE", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(img_with_text, "ESC pour annuler", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.imshow(sel_win, img_with_text)
                
                k2 = cv2.waitKey(1) & 0xFF

                # ========== SAISIE CLAVIER DU NUMERO ==========
                if k2 >= ord('0') and k2 <= ord('9'):
                    # Convertir la touche en index
                    input_index = k2 - ord('0')
                    
                    if 0 <= input_index < len(boxes):
                        target_box = boxes[input_index]
                        athlete_locked = True
                        frames_without_detection = 0
                        history_boxes = [target_box]
                        print(f"[INFO] Athlète sélectionné: index={input_index}, box={target_box}")
                        paused = False
                    else:
                        print(f"[WARN] Numéro {input_index} invalide. Cadres disponibles: 0-{len(boxes)-1}")

                # ESC pour annuler
                if k2 == 27:
                    print("[INFO] Sélection annulée.")
                    paused = False

                # Vérifier aussi fermeture de la fenêtre de sélection
                if cv2.getWindowProperty(sel_win, cv2.WND_PROP_VISIBLE) < 1:
                    paused = False

            cv2.destroyWindow(sel_win)

        # R pour reset
        if key == ord('r'):
            athlete_locked = False
            target_box = None
            history_boxes = []
            frames_without_detection = 0
            print("[INFO] Reset: plus de focus athlète.")

        tempsTotalFrame += fin - debut
        nbTotalFrame += 1

    # ===== ANALYSE OFFLINE DU SQUAT À LA FIN DE LA VIDÉO =====
    # Si c'est un squat, faire l'analyse complète à la fin
    if traitementVideo == affichageOrienteSquat and mouvement == IndiceMouvement.SQUAT.value:
        print("\n" + "="*60)
        print("ANALYSE OFFLINE DU SQUAT")
        print("="*60)
        
        # Analyser la séquence collectée
        from traitementVideo import analyze_squat_offline, display_squat_analysis
        squat_result = analyze_squat_offline(fps=fps)
        
        print(f"Verdict: {squat_result.get('verdict', 'INCONNU')}")
        print(f"Défaut: {squat_result.get('defaut', 'Aucun')}")
        print(f"Message: {squat_result.get('message', '')}")
        print("="*60 + "\n")
        
        # Créer une image finale avec les résultats (affichage à côté de la vidéo)
        try:
            H_img = image.shape[0]
            panel_w = max(400, int(H_img * 0.5))
            panel = np.ones((H_img, panel_w, 3), dtype=np.uint8) * 255
            display_squat_analysis(panel, squat_result, x_offset=10, y_offset=30)
            # Combiner côte-à-côte
            combined = np.hstack([cv2.resize(image, (image.shape[1], H_img)), panel])
            cv2.namedWindow("Analyse Finale", cv2.WINDOW_NORMAL)
            cv2.imshow("Analyse Finale", combined)
            print("Affichage de l'analyse finale. Appuyez sur une touche pour fermer.")
            cv2.waitKey(0)
            cv2.destroyWindow("Analyse Finale")
        except Exception as e:
            print(f"[WARN] Impossible d'afficher l'analyse finale: {e}")
    
    capture.release()
    cv2.destroyWindow(winName)
    if mediaPipe and modelMP is not None:
        modelMP.close()

    incrementerIndice(mouvement)

    vitesseMoyenne = calculerVitessePixels(trajectoire, fps)
    distanceParcourue = calculerDistancePixels(trajectoire)
    print("Vitesse moyenne en px/s : ", vitesseMoyenne)
    print("Distance parcourue en pixel : ", distanceParcourue)
    print("Traitement moyen d'un frame : ", tempsTotalFrame / max(1, nbTotalFrame))
    return tempsTotalFrame / max(1, nbTotalFrame)
