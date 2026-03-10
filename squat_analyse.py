import sys
import os
import math
from typing import List, Tuple, Optional
from collections import deque
from statistics import median

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QPixmap, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QLineEdit,
    QFormLayout,
    QGroupBox,
    QStatusBar,
)

# ============================================================
# PARAMÈTRES COMMUNS
# ============================================================

DEBUG_PIEDS = True
DEBUG_PIEDS_EVERY = 5

# Paramètres du détecteur robuste (Optical Flow)
FOOT_CALIBRATION_FRAMES = 30
FOOT_FLOW_ACCUMULATION_WINDOW = 5     # nombre de frames pour accumuler le déplacement
FOOT_DISPLACEMENT_THRESHOLD = 15.0    # seuil de déplacement en pixels
FOOT_PERSISTANCE_FRAMES = 8           # frames consécutives pour déclencher faute
FOOT_GRACE_PERIOD_FRAMES = 50         # frames supplémentaires après 'termine' pour détecter une faute tardive

SEUIL_VITESSE_RATIO_LARGEUR_BASSIN = 0.008
FENETRE_LISSAGE = 5
NB_IMAGES_CONSEC_DESCENTE = 4
NB_IMAGES_CONSEC_REMONTEE = 4
DUREE_MIN_DESCENTE_A_60_IPS = 6

SEUIL_VERROUILLAGE_GENOU = 173.0
DUREE_VERROUILLAGE_SEC = 0.12
BANDE_HAUT_RATIO_LARGEUR_BASSIN = 0.035
DUREE_HAUT_SEC = 0.12
SEUIL_STABILITE_RATIO_LARGEUR_BASSIN = 0.006
DELAI_MIN_AVANT_FIN_SEC = 0.25

SEUIL_ANGLE_GENOU_PROFONDEUR = 90.0

K_MIN_DIP_SEC = 0.06
X_IGNORE_AFTER_ASCENT_SEC = 0.05
A_MIN_DIP_RATIO_OF_HIPWIDTH = 0.015

CALIBRATION_FRAMES = 30
RAYON_TOLERANCE_RATIO = 0.10
SEUIL_MIN_PIXELS = 6.0
NB_CONSECUTIF_PIED_AVANT = 6

LARGEUR_PANNEAU = 680
COULEUR_FOND_PANNEAU_BGR = (28, 28, 28)

HANCHE_G, HANCHE_D = 11, 12
GENOU_G, GENOU_D = 13, 14
CHEVILLE_G, CHEVILLE_D = 15, 16

SQUELETTE_COCO17 = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def dessiner_squelette(image, points: List[Tuple[float, float]]) -> None:
    if not points or len(points) < 17:
        return

    for idx, (x, y) in enumerate(points):
        if x is None or y is None or (x == 0 and y == 0):
            continue
        cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(
            image, str(idx), (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
        )

    for a, b in SQUELETTE_COCO17:
        xa, ya = points[a]
        xb, yb = points[b]
        if (xa == 0 and ya == 0) or (xb == 0 and yb == 0):
            continue
        cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

    xg, yg = points[CHEVILLE_G]
    xd, yd = points[CHEVILLE_D]
    if not ((xg == 0 and yg == 0) or (xd == 0 and yd == 0)):
        cv2.line(image, (int(xg), int(yg)), (int(xd), int(yd)), (0, 255, 255), 2)


def choisir_personne_principale(result) -> Optional[List[Tuple[float, float]]]:
    if result is None or result.keypoints is None or result.keypoints.xy is None:
        return None

    kps_xy = result.keypoints.xy
    if len(kps_xy) == 0:
        return None

    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
        person = kps_xy[0].cpu().numpy().tolist()
        return [(float(x), float(y)) for x, y in person]

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(areas.argmax())

    person = kps_xy[best_idx].cpu().numpy().tolist()
    return [(float(x), float(y)) for x, y in person]


def limiter(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_degres(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    if (ax == 0 and ay == 0) or (bx == 0 and by == 0) or (cx == 0 and cy == 0):
        return None

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    norm_ba = math.hypot(ba[0], ba[1])
    norm_bc = math.hypot(bc[0], bc[1])
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosang = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cosang = limiter(cosang, -1.0, 1.0)
    return float(math.degrees(math.acos(cosang)))


def y_bassin_et_largeur(points: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    xg, yg = points[HANCHE_G]
    xd, yd = points[HANCHE_D]
    if (xg == 0 and yg == 0) or (xd == 0 and yd == 0):
        return None, None

    y_milieu = 0.5 * (yg + yd)
    largeur = math.hypot(xg - xd, yg - yd)
    if largeur < 1.0:
        largeur = None
    return float(y_milieu), float(largeur) if largeur is not None else None


def angle_moyen_genoux(points: List[Tuple[float, float]]) -> Optional[float]:
    g = angle_degres(points[HANCHE_G], points[GENOU_G], points[CHEVILLE_G])
    d = angle_degres(points[HANCHE_D], points[GENOU_D], points[CHEVILLE_D])
    vals = [v for v in (g, d) if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def y_moyenne_genoux(points: List[Tuple[float, float]]) -> Optional[float]:
    gx, gy = points[GENOU_G]
    dx, dy = points[GENOU_D]
    if (gx == 0 and gy == 0) or (dx == 0 and dy == 0):
        return None
    return float(0.5 * (gy + dy))


def trouver_police_ttf() -> Optional[str]:
    candidats = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for p in candidats:
        if os.path.exists(p):
            return p
    return None


def construire_polices(hauteur: int):
    chemin = trouver_police_ttf()
    if hauteur >= 900:
        taille_texte, taille_titre = 22, 28
    elif hauteur >= 720:
        taille_texte, taille_titre = 20, 26
    else:
        taille_texte, taille_titre = 18, 24

    if chemin:
        police_titre = ImageFont.truetype(chemin, taille_titre)
        police_texte = ImageFont.truetype(chemin, taille_texte)
        return police_titre, police_texte, chemin
    return ImageFont.load_default(), ImageFont.load_default(), None


def panneau_unicode(hauteur: int, lignes: List[str], titre: str, cache_polices: dict) -> np.ndarray:
    if hauteur not in cache_polices:
        cache_polices[hauteur] = construire_polices(hauteur)

    police_titre, police_texte, chemin_police = cache_polices[hauteur]

    panel_bgr = np.zeros((hauteur, LARGEUR_PANNEAU, 3), dtype=np.uint8)
    panel_bgr[:] = COULEUR_FOND_PANNEAU_BGR
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(panel_rgb)
    draw = ImageDraw.Draw(img)

    x = 16
    y = 12
    draw.text((x, y), titre, font=police_titre, fill=(255, 255, 255))
    y += 40

    draw.line((x, y + 20, LARGEUR_PANNEAU - 16, y + 20), fill=(100, 100, 100), width=2)
    y += 14

    if chemin_police is None:
        draw.text((x, y), "⚠ Police TTF non trouvée : installe DejaVu pour accents.", font=police_texte, fill=(255, 200, 120))
        y += 28

    marge = 8
    for s in lignes:
        if y > hauteur - 24:
            break
        draw.text((x, y), s, font=police_texte, fill=(235, 235, 235))
        bbox = draw.textbbox((x, y), s, font=police_texte)
        y += (bbox[3] - bbox[1]) + marge

    out_rgb = np.array(img)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class DetecteurPiedsRobuste:
    """
    Détecteur de déplacement des pieds basé sur le suivi optique (Optical Flow).
    Plus robuste que la détection par seules coordonnées YOLO.
    """
    def __init__(
        self,
        nb_images_calibration: int = 30,
        flow_accumulation_window: int = 5,
        deplacement_threshold_px: float = 15.0,
        nb_images_persistance: int = 8,
        grace_period_frames: int = 10,
        ignorer_apres_fin: bool = True,
        debug: bool = False,
    ):
        self.nb_images_calibration = nb_images_calibration
        self.flow_accumulation_window = flow_accumulation_window
        self.deplacement_threshold_px = deplacement_threshold_px
        self.nb_images_persistance = nb_images_persistance
        self.grace_period_frames = grace_period_frames
        self.ignorer_apres_fin = ignorer_apres_fin
        self.debug = debug

        # Calibration
        self.calibration_faite = False
        self._calib_xg, self._calib_xd = [], []
        self.ref_xg_init = None
        self.ref_xd_init = None

        # Tracking optique
        self.prev_gray = None
        self.prev_center_g = None
        self.prev_center_d = None
        self.displacement_buffer_g = deque(maxlen=flow_accumulation_window)
        self.displacement_buffer_d = deque(maxlen=flow_accumulation_window)

        # Détection de faute
        self.compteur_hors_seuil = 0
        self.faute = False
        
        # Période de grâce après 'termine'
        self.frame_numero_termine = None

    @staticmethod
    def _median(vals):
        return float(np.median(vals)) if len(vals) else 0.0

    def reset(self):
        """Réinitialise pour une nouvelle session."""
        self.__init__(
            nb_images_calibration=self.nb_images_calibration,
            flow_accumulation_window=self.flow_accumulation_window,
            deplacement_threshold_px=self.deplacement_threshold_px,
            nb_images_persistance=self.nb_images_persistance,
            grace_period_frames=self.grace_period_frames,
            ignorer_apres_fin=self.ignorer_apres_fin,
            debug=self.debug,
        )

    def update(
        self,
        frame: np.ndarray,
        etat: str,
        points,
        indice_image: int,
        ajouter_evenement=None,
    ) -> bool:
        """
        Met à jour l'état du détecteur et retourne True si faute détectée.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phase de calibration
        if not self.calibration_faite:
            if etat == "attente" and points is not None:
                xg, yg = points[CHEVILLE_G]
                xd, yd = points[CHEVILLE_D]
                if not ((xg == 0 and yg == 0) or (xd == 0 and yd == 0)):
                    self._calib_xg.append(float(xg))
                    self._calib_xd.append(float(xd))
                    if DEBUG_PIEDS and indice_image % DEBUG_PIEDS_EVERY == 0:
                        print(f"[PIEDS-CALIB] frame={indice_image} xg={xg:.1f} xd={xd:.1f} nb={len(self._calib_xg)}/{self.nb_images_calibration}")

                    if len(self._calib_xg) >= self.nb_images_calibration:
                        self.ref_xg_init = self._median(self._calib_xg)
                        self.ref_xd_init = self._median(self._calib_xd)
                        self.calibration_faite = True
                        self.prev_center_g = (int(self.ref_xg_init), int(h * 0.7))
                        self.prev_center_d = (int(self.ref_xd_init), int(h * 0.7))
                        if DEBUG_PIEDS:
                            print(f"[PIEDS-CALIB-OK] frame={indice_image} ref_xg={self.ref_xg_init:.1f} ref_xd={self.ref_xd_init:.1f}")
                        if ajouter_evenement:
                            ajouter_evenement(indice_image, "Calibration des pieds OK (suivi optique)")
            self.prev_gray = gray.copy()
            return self.faute

        # Mémoriser le passage à "termine" pour la fenêtre de grâce
        if etat == "termine" and self.frame_numero_termine is None:
            self.frame_numero_termine = indice_image

        # Arrêter après la période de grâce
        if self.ignorer_apres_fin and self.frame_numero_termine is not None:
            if indice_image > self.frame_numero_termine + self.grace_period_frames:
                self.prev_gray = gray.copy()
                return self.faute

        # Ne tracker que pendant descente, remontée et période de grâce
        if etat not in ("descente", "remontee", "termine"):
            self.compteur_hors_seuil = 0
            self.prev_gray = gray.copy()
            return self.faute

        # Tracking optique Lucas-Kanade
        if self.prev_gray is not None and self.prev_center_g is not None and self.prev_center_d is not None:
            pts_g = np.array([[self.prev_center_g]], dtype=np.float32)
            pts_d = np.array([[self.prev_center_d]], dtype=np.float32)

            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            try:
                pts_g_new, status_g, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, pts_g, None, **lk_params
                )
                pts_d_new, status_d, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, pts_d, None, **lk_params
                )

                center_g_new = (int(pts_g_new[0, 0, 0]), int(pts_g_new[0, 0, 1])) if status_g is not None and status_g[0] else self.prev_center_g
                center_d_new = (int(pts_d_new[0, 0, 0]), int(pts_d_new[0, 0, 1])) if status_d is not None and status_d[0] else self.prev_center_d

                disp_g = math.hypot(
                    center_g_new[0] - self.prev_center_g[0],
                    center_g_new[1] - self.prev_center_g[1],
                )
                disp_d = math.hypot(
                    center_d_new[0] - self.prev_center_d[0],
                    center_d_new[1] - self.prev_center_d[1],
                )

                self.displacement_buffer_g.append(disp_g)
                self.displacement_buffer_d.append(disp_d)

                cumul_g = sum(self.displacement_buffer_g)
                cumul_d = sum(self.displacement_buffer_d)
                score = max(cumul_g, cumul_d)

                if DEBUG_PIEDS and indice_image % DEBUG_PIEDS_EVERY == 0:
                    print(f"[PIEDS-FLOW] frame={indice_image} etat={etat} disp_g={disp_g:.1f} disp_d={disp_d:.1f} cumul_g={cumul_g:.1f} cumul_d={cumul_d:.1f} score={score:.1f} seuil={self.deplacement_threshold_px:.1f} compteur={self.compteur_hors_seuil}/{self.nb_images_persistance}")

                if score > self.deplacement_threshold_px:
                    self.compteur_hors_seuil += 1
                else:
                    self.compteur_hors_seuil = 0

                if (not self.faute) and self.compteur_hors_seuil >= self.nb_images_persistance:
                    self.faute = True
                    if DEBUG_PIEDS:
                        print(f"[PIEDS-FAUTE] frame={indice_image} score={score:.1f} seuil={self.deplacement_threshold_px:.1f} compteur={self.compteur_hors_seuil}/{self.nb_images_persistance}")
                    if ajouter_evenement:
                        ajouter_evenement(indice_image, f"FAUTE : déplacement des pieds détecté (score={score:.1f}px > seuil={self.deplacement_threshold_px:.1f}px)")

                self.prev_center_g = center_g_new
                self.prev_center_d = center_d_new

            except Exception as e:
                if DEBUG_PIEDS:
                    print(f"[PIEDS-ERROR] frame={indice_image} {str(e)}")

        self.prev_gray = gray.copy()
        return self.faute


class VideoWorker(QThread):
    image_ready = Signal(QImage)
    status_ready = Signal(str)
    finished_cleanly = Signal()
    error_signal = Signal(str)

    def __init__(self, video_path: str, model_path: str, conf: float):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.conf = conf
        #self.device = device or None
        self._pause = False
        self._stop = False
        self._restart = False

    def pause_toggle(self):
        self._pause = not self._pause

    def request_stop(self):
        self._stop = True

    def request_restart(self):
        self._restart = True

    def _emit_status(self, msg: str):
        self.status_ready.emit(msg)

    def run(self):
        try:
            if not os.path.exists(self.video_path):
                raise FileNotFoundError(f"Vidéo introuvable: {self.video_path}")

            modele = YOLO(self.model_path)
            cache_polices = {}

            while not self._stop:
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    raise RuntimeError("Impossible d'ouvrir la vidéo.")

                largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                images_par_seconde = float(cap.get(cv2.CAP_PROP_FPS))
                if images_par_seconde <= 1e-6:
                    images_par_seconde = 30.0

                duree_min_descente_images = max(2, int(round(DUREE_MIN_DESCENTE_A_60_IPS * (images_par_seconde / 60.0))))
                verrouillage_images = max(2, int(round(DUREE_VERROUILLAGE_SEC * images_par_seconde)))
                maintien_haut_images = max(2, int(round(DUREE_HAUT_SEC * images_par_seconde)))
                delai_min_avant_fin_images = max(0, int(round(DELAI_MIN_AVANT_FIN_SEC * images_par_seconde)))
                k_min_dip_frames = max(2, int(round(K_MIN_DIP_SEC * images_par_seconde)))
                x_ignore_frames = max(0, int(round(X_IGNORE_AFTER_ASCENT_SEC * images_par_seconde)))

                buffer_y_bassin = deque(maxlen=FENETRE_LISSAGE)
                buffer_largeur_bassin = deque(maxlen=FENETRE_LISSAGE)
                buffer_angle_genou = deque(maxlen=FENETRE_LISSAGE)
                buffer_position_haute = deque(maxlen=int(max(10, round(1.0 * images_par_seconde))))
                journal = deque(maxlen=15)

                def ajouter_evenement(image_i: int, message: str):
                    t = image_i / images_par_seconde if image_i >= 0 else 0.0
                    journal.appendleft(f"[{t:5.2f}s] {message}")
                    self._emit_status(message)

                etat = "attente"
                indice_image = -1
                suite_descente = 0
                suite_remontee = 0
                y_bassin_lisse_avant = None

                image_debut_descente = None
                image_debut_remontee = None
                image_fin_remontee = None
                y_position_haute = None

                y_bassin_max = None
                image_point_bas = None
                angle_genou_point_bas = None
                y_genoux_point_bas = None
                faute_descente_insuffisante = False
                hanches_sous_genoux = None
                angle_genou_ok = None
                angle_genou_min_observe = None

                dip_detected = False
                dip_start_frame = None
                dip_amp_px = None
                dip_streak = 0
                dip_candidate_start = None
                dip_base_y = None
                dip_peak_y = None

                suite_verrouillage = 0
                suite_haut = 0
                detecteur_pieds = DetecteurPiedsRobuste(
                    nb_images_calibration=FOOT_CALIBRATION_FRAMES,
                    flow_accumulation_window=FOOT_FLOW_ACCUMULATION_WINDOW,
                    deplacement_threshold_px=FOOT_DISPLACEMENT_THRESHOLD,
                    nb_images_persistance=FOOT_PERSISTANCE_FRAMES,
                    grace_period_frames=FOOT_GRACE_PERIOD_FRAMES,
                    ignorer_apres_fin=True,
                    debug=DEBUG_PIEDS,
                )

                ajouter_evenement(-1, "Analyse démarrée")

                while not self._stop:
                    if self._restart:
                        self._restart = False
                        break

                    if self._pause:
                        self.msleep(30)
                        continue

                    ok, frame = cap.read()
                    if not ok:
                        self._emit_status("Fin de la vidéo")
                        self.finished_cleanly.emit()
                        cap.release()
                        return

                    indice_image += 1
                    video_propre = frame.copy()

                    resultats = modele.predict(source=video_propre, conf=self.conf, verbose=False)
                    r0 = resultats[0] if resultats else None

                    points = choisir_personne_principale(r0)
                    if points is not None:
                        dessiner_squelette(video_propre, points)

                        y_bassin, largeur_bassin = y_bassin_et_largeur(points)
                        if y_bassin is not None:
                            buffer_y_bassin.append(y_bassin)
                            buffer_position_haute.append(y_bassin)
                            if largeur_bassin is not None:
                                buffer_largeur_bassin.append(largeur_bassin)

                        angle_genou = angle_moyen_genoux(points)
                        if angle_genou is not None:
                            buffer_angle_genou.append(angle_genou)

                    y_bassin_lisse = None
                    largeur_bassin_lisse = None
                    angle_genou_lisse = None
                    vitesse = None

                    if len(buffer_y_bassin) >= max(3, FENETRE_LISSAGE // 2):
                        y_bassin_lisse = float(median(buffer_y_bassin))
                        largeur_bassin_lisse = float(median(buffer_largeur_bassin)) if len(buffer_largeur_bassin) > 0 else 200.0
                        angle_genou_lisse = float(median(buffer_angle_genou)) if len(buffer_angle_genou) > 0 else None

                        if angle_genou_lisse is not None:
                            if angle_genou_min_observe is None or angle_genou_lisse < angle_genou_min_observe:
                                angle_genou_min_observe = angle_genou_lisse

                        seuil_vitesse_px = SEUIL_VITESSE_RATIO_LARGEUR_BASSIN * largeur_bassin_lisse
                        seuil_stabilite_px = SEUIL_STABILITE_RATIO_LARGEUR_BASSIN * largeur_bassin_lisse
                        bande_haut_px = BANDE_HAUT_RATIO_LARGEUR_BASSIN * largeur_bassin_lisse
                        a_min_dip_px = A_MIN_DIP_RATIO_OF_HIPWIDTH * largeur_bassin_lisse

                        if y_bassin_lisse_avant is not None:
                            vitesse = y_bassin_lisse - y_bassin_lisse_avant
                            suite_descente = suite_descente + 1 if vitesse > seuil_vitesse_px else 0
                            suite_remontee = suite_remontee + 1 if vitesse < -seuil_vitesse_px else 0

                            if etat == "attente":
                                if suite_descente >= NB_IMAGES_CONSEC_DESCENTE:
                                    etat = "descente"
                                    image_debut_descente = indice_image - NB_IMAGES_CONSEC_DESCENTE + 1
                                    y_position_haute = float(median(buffer_position_haute)) if buffer_position_haute else y_bassin_lisse_avant
                                    y_bassin_max = y_bassin_lisse
                                    image_point_bas = indice_image
                                    angle_genou_point_bas = angle_genou_lisse
                                    y_genoux_point_bas = y_moyenne_genoux(points) if points is not None else None
                                    ajouter_evenement(indice_image, f"Début de descente (frame {image_debut_descente})")

                            elif etat == "descente":
                                if y_bassin_max is None or y_bassin_lisse > y_bassin_max:
                                    y_bassin_max = y_bassin_lisse
                                    image_point_bas = indice_image
                                    angle_genou_point_bas = angle_genou_lisse
                                    y_genoux_point_bas = y_moyenne_genoux(points) if points is not None else None

                                if image_debut_descente is not None and (indice_image - image_debut_descente) >= duree_min_descente_images:
                                    if suite_remontee >= NB_IMAGES_CONSEC_REMONTEE:
                                        etat = "remontee"
                                        image_debut_remontee = indice_image - NB_IMAGES_CONSEC_REMONTEE + 1
                                        ajouter_evenement(indice_image, f"Début de remontée (frame {image_debut_remontee})")

                                        if y_bassin_max is not None:
                                            if y_genoux_point_bas is not None:
                                                hanches_sous_genoux = (y_bassin_max > y_genoux_point_bas)
                                            else:
                                                hanches_sous_genoux = None

                                            if angle_genou_point_bas is not None:
                                                angle_genou_ok = (angle_genou_point_bas < SEUIL_ANGLE_GENOU_PROFONDEUR)
                                            else:
                                                angle_genou_ok = None

                                            profondeur_ok = (hanches_sous_genoux is True) or (angle_genou_ok is True)
                                            faute_descente_insuffisante = not profondeur_ok
                                            ajouter_evenement(indice_image, "FAUTE : descente non suffisante" if faute_descente_insuffisante else "OK : descente suffisante")

                                        dip_streak = 0
                                        dip_candidate_start = None
                                        dip_base_y = None
                                        dip_peak_y = None
                                        suite_verrouillage = 0
                                        suite_haut = 0

                            elif etat == "remontee":
                                if image_debut_remontee is not None and (indice_image - image_debut_remontee) >= delai_min_avant_fin_images:
                                    stable = abs(vitesse) < seuil_stabilite_px

                                    if angle_genou_lisse is not None and angle_genou_lisse >= SEUIL_VERROUILLAGE_GENOU and stable:
                                        suite_verrouillage += 1
                                    else:
                                        suite_verrouillage = 0

                                    if y_position_haute is not None and abs(y_bassin_lisse - y_position_haute) <= bande_haut_px and stable:
                                        suite_haut += 1
                                    else:
                                        suite_haut = 0

                                    if suite_verrouillage >= verrouillage_images or suite_haut >= maintien_haut_images:
                                        etat = "termine"
                                        image_fin_remontee = indice_image
                                        raison = "genoux verrouillés" if suite_verrouillage >= verrouillage_images else "retour en haut"
                                        ajouter_evenement(indice_image, f"Fin de remontée ({raison}, frame {image_fin_remontee})")

                                if not dip_detected and image_debut_remontee is not None and (indice_image - image_debut_remontee) >= x_ignore_frames:
                                    if vitesse > seuil_vitesse_px:
                                        if dip_candidate_start is None:
                                            dip_candidate_start = indice_image
                                            dip_base_y = y_bassin_lisse_avant
                                            dip_peak_y = y_bassin_lisse
                                        dip_streak += 1
                                        dip_peak_y = max(dip_peak_y, y_bassin_lisse)
                                    else:
                                        if dip_candidate_start is not None:
                                            amp = (dip_peak_y - dip_base_y) if (dip_peak_y is not None and dip_base_y is not None) else 0.0
                                            if dip_streak >= k_min_dip_frames and amp >= a_min_dip_px:
                                                dip_detected = True
                                                dip_start_frame = dip_candidate_start
                                                dip_amp_px = amp
                                                ajouter_evenement(indice_image, f"FAUTE : redescente détectée (frame {dip_start_frame}, amplitude {amp:.1f}px)")
                                            dip_streak = 0
                                            dip_candidate_start = None
                                            dip_base_y = None
                                            dip_peak_y = None

                        y_bassin_lisse_avant = y_bassin_lisse

                    faute_pied_avant = detecteur_pieds.update(
                        frame=video_propre,
                        etat=etat,
                        points=points,
                        indice_image=indice_image,
                        ajouter_evenement=ajouter_evenement,
                    )

                    if DEBUG_PIEDS and indice_image % DEBUG_PIEDS_EVERY == 0:
                        yb = f"{y_bassin_lisse:.1f}" if y_bassin_lisse is not None else "—"
                        vit = f"{vitesse:+.2f}" if vitesse is not None else "—"
                        lb = f"{largeur_bassin_lisse:.1f}" if largeur_bassin_lisse is not None else "—"
                        ag = f"{angle_genou_lisse:.1f}" if angle_genou_lisse is not None else "—"
                        print(f"[GLOBAL] frame={indice_image} etat={etat} y_bassin={yb} vitesse={vit} largeur_bassin={lb} angle_genou={ag} faute_pieds={faute_pied_avant}")

                    t_sec = indice_image / images_par_seconde
                    ips_txt = f"{images_par_seconde:.2f} ips"
                    y_txt = f"{y_bassin_lisse:.1f}" if y_bassin_lisse is not None else "—"
                    largeur_txt = f"{largeur_bassin_lisse:.1f}px" if largeur_bassin_lisse is not None else "—"
                    vitesse_txt = f"{vitesse:+.2f}px" if vitesse is not None else "—"
                    angle_txt = f"{angle_genou_lisse:.1f}°" if angle_genou_lisse is not None else "—"
                    angle_min_txt = f"{angle_genou_min_observe:.1f}°" if angle_genou_min_observe is not None else "—"
                    angle_pb_txt = f"{angle_genou_point_bas:.1f}°" if angle_genou_point_bas is not None else "—"
                    verdict_profondeur = "—"
                    if etat in ("remontee", "termine") and image_debut_remontee is not None:
                        verdict_profondeur = "FAUTE" if faute_descente_insuffisante else "OK"

                    lignes = [
                        f"Cadence : {ips_txt}",
                        f"Image : {indice_image}    Temps : {t_sec:.2f} s",
                        f"Phase : {etat}",
                        "",
                        "=== REPÈRES TEMPORELS ===",
                        f"Début descente : {image_debut_descente if image_debut_descente is not None else '—'}",
                        f"Début remontée : {image_debut_remontee if image_debut_remontee is not None else '—'}",
                        f"Fin remontée   : {image_fin_remontee if image_fin_remontee is not None else '—'}",
                        "",
                        "=== DONNÉES LISSÉES ===",
                        f"Y bassin (px)        : {y_txt}",
                        f"Largeur bassin (px)  : {largeur_txt}",
                        f"Vitesse bassin (px/f): {vitesse_txt}",
                        f"Angle genou (°)      : {angle_txt}",
                        f"Angle genou min obs. : {angle_min_txt}",
                        "",
                        "=== ANALYSE PROFONDEUR ===",
                        f"Image point bas      : {image_point_bas if image_point_bas is not None else '—'}",
                        f"Angle genou au point bas : {angle_pb_txt}",
                        f"Hanches sous genoux  : {hanches_sous_genoux if hanches_sous_genoux is not None else '—'}",
                        f"Angle < {SEUIL_ANGLE_GENOU_PROFONDEUR:.0f}° : {angle_genou_ok if angle_genou_ok is not None else '—'}",
                        f"Verdict profondeur   : {verdict_profondeur}",
                        "",
                        "=== ANALYSE REDESCENTE (DIP) ===",
                        f"Dip détecté ?        : {'Détecté' if dip_detected else 'Non détecté'}",
                        f"Frame début dip      : {dip_start_frame if dip_start_frame is not None else '—'}",
                        f"Amplitude dip (px)   : {dip_amp_px:.1f}" if dip_amp_px is not None else "Amplitude dip (px)   : —",
                        "",
                        "=== ANALYSE PIEDS ===",
                        f"Calibration OK       : {'Oui' if detecteur_pieds.calibration_faite else 'Non (en cours)'}",
                        f"État                 : {'FAUTE' if faute_pied_avant else 'OK'}",
                        f"Compteur hors seuil  : {detecteur_pieds.compteur_hors_seuil}/{detecteur_pieds.nb_images_persistance}",
                        "",
                        "=== ÉVÉNEMENTS RÉCENTS ===",
                    ]
                    lignes.extend(list(journal))

                    panneau = panneau_unicode(hauteur, lignes, "TABLEAU DE BORD — SQUAT", cache_polices)
                    combo = np.hstack([video_propre, panneau])
                    self.image_ready.emit(frame_to_qimage(combo))
                    self.msleep(max(1, int(1000 / images_par_seconde)))

                cap.release()

        except Exception as e:
            self.error_signal.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyse powerlifting — PySide6")
        self.resize(1650, 950)
        self.worker = None
        self.video_path = ""
        self._build_ui()
        self._build_menu()
        self._apply_style()

    def _build_ui(self):
        from PySide6.QtWidgets import QComboBox

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 3)

        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Choisir une vidéo…")

        self.model_edit = QLineEdit("yolov8n-pose.pt")
        '''self.device_edit = QLineEdit()
        self.device_edit.setPlaceholderText("cpu / cuda / mps")'''

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.25)

        self.mouvement_combo = QComboBox()
        self.mouvement_combo.addItems([
            "Squat",
            "Développé couché",
            "Soulevé de terre",
        ])

        self.vue_combo = QComboBox()
        self.vue_combo.addItems([
            "1 vue (face seulement)",
            "2 vues",
            "3 vues (face + 2 latérales)",
        ])

        form_box = QGroupBox("Paramètres")
        form = QFormLayout(form_box)
        form.addRow("Type de mouvement", self.mouvement_combo)
        form.addRow("Nombre de vues", self.vue_combo)
        form.addRow("Vidéo", self.video_edit)
        form.addRow("Modèle", self.model_edit)
        form.addRow("Confiance", self.conf_spin)
        '''form.addRow("Device", self.device_edit)'''

        self.info_label = QLabel(
            "Choisis d'abord le mouvement et le nombre de vues."
            "- Squat : moteur déjà branché"
            "- Développé couché : placeholder"
            "- Soulevé de terre : placeholder"
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding:10px; background:#25272c; border:1px solid #3f4248; border-radius:10px;")

        self.btn_open = QPushButton("Ouvrir une vidéo")
        self.btn_start = QPushButton("Lancer")
        self.btn_pause = QPushButton("Pause / Reprendre")
        self.btn_restart = QPushButton("Recommencer")
        self.btn_stop = QPushButton("Arrêter")

        self.btn_open.clicked.connect(self.open_video)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_restart.clicked.connect(self.restart_analysis)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.mouvement_combo.currentTextChanged.connect(self.update_mode_info)
        self.vue_combo.currentTextChanged.connect(self.update_mode_info)

        left.addWidget(form_box)
        left.addWidget(self.info_label)
        left.addWidget(self.btn_open)
        left.addWidget(self.btn_start)
        left.addWidget(self.btn_pause)
        left.addWidget(self.btn_restart)
        left.addWidget(self.btn_stop)
        left.addStretch(1)

        self.image_label = QLabel("Aucune vidéo chargée")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setStyleSheet("background:#111; border:1px solid #333; border-radius:12px;")
        right.addWidget(self.image_label)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Prêt")
        self.update_mode_info()

    def update_mode_info(self):
        mouvement = self.mouvement_combo.currentText()
        vues = self.vue_combo.currentText()
        if mouvement == "Squat":
            txt = (
                f"Mode actuel : {mouvement}"
                f"Configuration vidéo : {vues}"
                "Le moteur squat est déjà connecté."
                "Plus tard, tu pourras brancher un moteur différent pour le développé couché et un autre pour le soulevé de terre."
            )
        elif mouvement == "Développé couché":
            txt = (
                f"Mode actuel : {mouvement}"
                f"Configuration vidéo : {vues}"
                "Le moteur développé couché n'est pas encore implémenté."
                "L'interface est déjà prête pour le recevoir."
            )
        else:
            txt = (
                f"Mode actuel : {mouvement}"
                f"Configuration vidéo : {vues}"
                "Le moteur soulevé de terre n'est pas encore implémenté."
                "L'interface est déjà prête pour le recevoir."
            )
        self.info_label.setText(txt)

    def _build_menu(self):
        menu = self.menuBar().addMenu("Fichier")
        open_action = QAction("Ouvrir", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_video)
        menu.addAction(open_action)

        quit_action = QAction("Quitter", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #1e1f22; color: #f3f3f3; }
            QLabel, QGroupBox { color: #f3f3f3; }
            QLineEdit, QDoubleSpinBox, QComboBox {
                background: #2b2d31;
                color: #f3f3f3;
                border: 1px solid #3f4248;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton {
                background: #3a7afe;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #5b90ff; }
            QPushButton:pressed { background: #2e65d3; }
            QGroupBox {
                border: 1px solid #3f4248;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 12px;
                background: #25272c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QStatusBar { background: #25272c; color: #d7d7d7; }
            """
        )

    @Slot()
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une vidéo",
            "",
            "Vidéos (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;Tous les fichiers (*.*)",
        )
        if path:
            self.video_path = path
            self.video_edit.setText(path)
            self.statusBar().showMessage(f"Vidéo chargée : {os.path.basename(path)}")

    @Slot()
    def start_analysis(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Analyse en cours", "Arrête l'analyse actuelle avant d'en lancer une autre.")
            return

        video_path = self.video_edit.text().strip()
        if not video_path:
            QMessageBox.warning(self, "Vidéo manquante", "Choisis d'abord une vidéo.")
            return

        mouvement = self.mouvement_combo.currentText()
        nb_vues = self.vue_combo.currentText()

        if mouvement != "Squat":
            QMessageBox.information(
                self,
                "Moteur non encore disponible",
                f"Le mode '{mouvement}' est sélectionné avec '{nb_vues}', mais le moteur correspondant n'est pas encore codé."
                "L'interface est déjà prête. Il suffira ensuite de brancher le bon fichier de traitement selon le mouvement choisi."
            )
            return

        self.worker = VideoWorker(
            video_path=video_path,
            model_path=self.model_edit.text().strip() or "yolov8n-pose.pt",
            conf=float(self.conf_spin.value())
            #device=self.device_edit.text().strip() or None,
        )
        self.worker.image_ready.connect(self.update_image)
        self.worker.status_ready.connect(self.statusBar().showMessage)
        self.worker.error_signal.connect(self.show_error)
        self.worker.finished_cleanly.connect(lambda: self.statusBar().showMessage("Analyse terminée"))
        self.worker.start()
        self.statusBar().showMessage(f"Analyse lancée — {mouvement} — {nb_vues}")

    @Slot()
    def toggle_pause(self):
        if self.worker and self.worker.isRunning():
            self.worker.pause_toggle()
            self.statusBar().showMessage("Pause/Reprise demandée")

    @Slot()
    def restart_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_restart()
            self.statusBar().showMessage("Redémarrage demandé")

    @Slot()
    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait()
            self.statusBar().showMessage("Analyse arrêtée")

    @Slot(QImage)
    def update_image(self, qimage: QImage):
        pix = QPixmap.fromImage(qimage)
        pix = pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pix)

    @Slot(str)
    def show_error(self, message: str):
        QMessageBox.critical(self, "Erreur", message)
        self.statusBar().showMessage("Erreur")

    def closeEvent(self, event):
        self.stop_analysis()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
