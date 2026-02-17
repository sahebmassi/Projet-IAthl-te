import cv2
from enumIndice import *
from traitementVideo import *
from trouverArticulationYolo import *
import time
from onnxOptimiser import *
from exportationDonneeCSV import exportationCSV, incrementerIndice

HAUTEUR = 480
LARGEUR = 640


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
    boxes = param  # liste de boxes affichées
    _clicked_point = (x, y)
    for i, (xMin, yMin, xMax, yMax) in enumerate(boxes):
        if xMin <= x <= xMax and yMin <= y <= yMax:
            _selected_index = i
            break

def _box_center(box):
    xMin, yMin, xMax, yMax = box
    return ((xMin + xMax) / 2.0, (yMin + yMax) / 2.0)

def _reorder_by_target(arts, boxes, target_box):
    """
    Réordonne arts/boxes pour mettre l'athlète en index 0.
    On choisit la box la plus proche du target_box (centre-centre).
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

    # Réordonner
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

    modelMP, modelYolo = None, yolo
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
    # Etat sélection ATHLETE
    # =============================
    athlete_locked = False
    target_box = None  # box actuelle de l'athlète (mise à jour frame par frame)

    paused = False

    while capture.isOpened():
        retour, image = capture.read()
        if not retour:
            print("Une erreur avec la capture de l'image.")
            break

        debut = time.time()

        # Détection YOLO (articulations + boxes)
        coordonnees = trouverArticulation(image, modelMP, modelYolo)
        if enregistrerVideoAvant is not None:
            outAvant.write(cv2.resize(image.copy(), (LARGEUR, HAUTEUR), interpolation=cv2.INTER_AREA))

        # coordonnees = (arts, boxes)
        arts, boxes = coordonnees[0], coordonnees[1]

        # Si athlète sélectionné => on force l'athlète en index 0
        if athlete_locked and boxes and arts:
            arts, boxes, target_box = _reorder_by_target(arts, boxes, target_box)

        coordonnees = (arts, boxes)
        touteCoordonnees.append(coordonnees)

        # --------- Trajectoire (si besoin) ----------
        if len(touteCoordonnees[-1][1]) > 0:
            xMin, yMin, xMax, yMax = touteCoordonnees[-1][1][0]
            if pov == IndicePointDeVue.GAUCHE.value:
                boutBarre = (xMin, int((yMin + yMax) / 2))
            elif pov == IndicePointDeVue.DROITE.value:
                boutBarre = (xMax, int((yMin + yMax) / 2))
            else:
                boutBarre = (int(xMin + (xMax - xMin) / 2), yMax)
            trajectoire.append(boutBarre)

        # Traitement métier (dessins / règles)
        mouvementReussi = traitementVideo(touteCoordonnees, image, winName)

        # Afficher image à résolution native (pas de resize)
        cv2.imshow(winName, image)

        if enregistrementJson is not None:
            exportationCSV(touteCoordonnees, enregistrementJson, mouvement)

        fin = time.time()
        if enregistrementVideoApres is not None:
            outApres.write(image)

        # =============================
        # Gestion clavier + Fermeture
        # =============================
        key = cv2.waitKey(20) & 0xFF

        # Quitter si touche 'q' OU si fenêtre fermée (clique sur X)
        if key == ord('q') or cv2.getWindowProperty(winName, cv2.WND_PROP_VISIBLE) < 1:
            break

        # ---- Appui sur V => PAUSE + Sélection ----
        if key == ord('v'):
            paused = True
            _selected_index = None
            _clicked_point = None

            # on prépare une image de sélection (en taille affichée)
            sel_img = cv2.resize(image.copy(), (LARGEUR, HAUTEUR), interpolation=cv2.INTER_AREA)

            # IMPORTANT: boxes sont en coords frame originale -> on scale vers la fenêtre
            H0, W0 = image.shape[:2]
            sx = LARGEUR / float(W0)
            sy = HAUTEUR / float(H0)

            scaled_boxes = []
            for (xMin, yMin, xMax, yMax) in boxes:
                scaled_boxes.append((
                    int(xMin * sx), int(yMin * sy),
                    int(xMax * sx), int(yMax * sy)
                ))

            _draw_numbered_boxes(sel_img, scaled_boxes)

            sel_win = f"Selection Athlete {id}"
            cv2.namedWindow(sel_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(sel_win, LARGEUR, HAUTEUR)
            # essayer de mettre la fenêtre à côté
            cv2.moveWindow(sel_win, 50 + LARGEUR, 50)
            cv2.setMouseCallback(sel_win, _click_select_box, scaled_boxes)

            while paused:
                cv2.imshow(sel_win, sel_img)
                k2 = cv2.waitKey(30) & 0xFF

                # Si clique valide
                if _selected_index is not None:
                    # target_box en coords originales
                    if 0 <= _selected_index < len(boxes):
                        target_box = boxes[_selected_index]
                        athlete_locked = True
                        print(f"[INFO] Athlète sélectionné: index={_selected_index}, box={target_box}")
                    paused = False

                # ESC pour annuler
                if k2 == 27:
                    print("[INFO] Sélection annulée.")
                    paused = False

                # Vérifier aussi fermeture de la fenêtre de sélection
                if cv2.getWindowProperty(sel_win, cv2.WND_PROP_VISIBLE) < 1:
                    paused = False

            cv2.destroyWindow(sel_win)

        # optionnel: R pour reset (retour comportement par défaut)
        if key == ord('r'):
            athlete_locked = False
            target_box = None
            print("[INFO] Reset: plus de focus athlète.")

        tempsTotalFrame += fin - debut
        nbTotalFrame += 1

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
