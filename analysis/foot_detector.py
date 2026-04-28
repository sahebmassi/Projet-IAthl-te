import math
"""Détection robuste des déplacements de pieds.

Le détecteur calibre les positions initiales des chevilles pendant l'état de
repos, puis surveille les écarts pendant les phases actives. Quand les keypoints
de cheville YOLO sont disponibles, ils sont utilisés en priorité. L'optical flow
sert seulement de secours.

Pour limiter les faux positifs en squat/deadlift, le déplacement vertical est
pondéré par vertical_weight: les chevilles peuvent monter/descendre dans l'image
à cause de la flexion, sans que le pied glisse réellement au sol.
"""

from collections import deque

import cv2
import numpy as np

from .constants import CHEVILLE_D, CHEVILLE_G, DEBUG_PIEDS, DEBUG_PIEDS_EVERY


class DetecteurPiedsRobuste:
    """
    DÃ©tecteur de dÃ©placement des pieds basÃ© sur le suivi optique (Optical Flow).
    Plus robuste que la dÃ©tection par seules coordonnÃ©es YOLO.
    """

    def __init__(
        self,
        nb_images_calibration: int = 30,
        flow_accumulation_window: int = 5,
        deplacement_threshold_px: float = 15.0,
        nb_images_persistance: int = 8,
        grace_period_frames: int = 10,
        vertical_weight: float = 1.0,
        ignorer_apres_fin: bool = True,
        debug: bool = False,
    ):
        self.nb_images_calibration = nb_images_calibration
        self.flow_accumulation_window = flow_accumulation_window
        self.deplacement_threshold_px = deplacement_threshold_px
        self.nb_images_persistance = nb_images_persistance
        self.grace_period_frames = grace_period_frames
        self.vertical_weight = vertical_weight
        self.ignorer_apres_fin = ignorer_apres_fin
        self.debug = debug

        self.calibration_faite = False
        self._calib_xg, self._calib_xd = [], []
        self._calib_yg, self._calib_yd = [], []
        self.ref_xg_init = None
        self.ref_xd_init = None
        self.ref_yg_init = None
        self.ref_yd_init = None

        self.prev_gray = None
        self.prev_center_g = None
        self.prev_center_d = None
        self.shift_buffer_g = deque(maxlen=flow_accumulation_window)
        self.shift_buffer_d = deque(maxlen=flow_accumulation_window)

        self.compteur_hors_seuil = 0
        self.faute = False
        self.frame_numero_termine = None

    @staticmethod
    def _median(vals):
        return float(np.median(vals)) if len(vals) else 0.0

    @staticmethod
    def _valid_ankles(points):
        if points is None:
            return None
        xg, yg = points[CHEVILLE_G]
        xd, yd = points[CHEVILLE_D]
        if (xg == 0 and yg == 0) or (xd == 0 and yd == 0):
            return None
        return (int(xg), int(yg)), (int(xd), int(yd))

    def reset(self):
        """Reset calibration, buffers and fault state while preserving settings."""

        self.__init__(
            nb_images_calibration=self.nb_images_calibration,
            flow_accumulation_window=self.flow_accumulation_window,
            deplacement_threshold_px=self.deplacement_threshold_px,
            nb_images_persistance=self.nb_images_persistance,
            grace_period_frames=self.grace_period_frames,
            vertical_weight=self.vertical_weight,
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
        calibration_states=None,
        active_states=None,
    ) -> bool:
        """Update the detector with one frame and return whether a fault exists.

        During calibration states, ankle references are collected. During active
        states, the detector compares current ankle positions to those
        references. A fault is raised only after enough persistent frames above
        the configured threshold.
        """

        if calibration_states is None:
            calibration_states = {"attente"}
        if active_states is None:
            active_states = {"descente", "remontee", "termine"}

        h, _ = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.calibration_faite:
            if etat in calibration_states and points is not None:
                xg, yg = points[CHEVILLE_G]
                xd, yd = points[CHEVILLE_D]
                if not ((xg == 0 and yg == 0) or (xd == 0 and yd == 0)):
                    self._calib_xg.append(float(xg))
                    self._calib_xd.append(float(xd))
                    self._calib_yg.append(float(yg))
                    self._calib_yd.append(float(yd))
                    if DEBUG_PIEDS and indice_image % DEBUG_PIEDS_EVERY == 0:
                        print(f"[PIEDS-CALIB] frame={indice_image} xg={xg:.1f} xd={xd:.1f} nb={len(self._calib_xg)}/{self.nb_images_calibration}")

                    if len(self._calib_xg) >= self.nb_images_calibration:
                        self.ref_xg_init = self._median(self._calib_xg)
                        self.ref_xd_init = self._median(self._calib_xd)
                        self.ref_yg_init = self._median(self._calib_yg)
                        self.ref_yd_init = self._median(self._calib_yd)
                        self.calibration_faite = True
                        self.prev_center_g = (
                            int(self.ref_xg_init),
                            int(min(max(self.ref_yg_init, 0.0), h - 1)),
                        )
                        self.prev_center_d = (
                            int(self.ref_xd_init),
                            int(min(max(self.ref_yd_init, 0.0), h - 1)),
                        )
                        if DEBUG_PIEDS:
                            print(f"[PIEDS-CALIB-OK] frame={indice_image} ref_xg={self.ref_xg_init:.1f} ref_xd={self.ref_xd_init:.1f}")
                        if ajouter_evenement:
                            ajouter_evenement(indice_image, "Calibration des pieds OK (suivi optique)")
            self.prev_gray = gray.copy()
            return self.faute

        if etat == "termine" and self.frame_numero_termine is None:
            self.frame_numero_termine = indice_image

        if self.ignorer_apres_fin and self.frame_numero_termine is not None:
            if indice_image > self.frame_numero_termine + self.grace_period_frames:
                self.prev_gray = gray.copy()
                return self.faute

        if etat not in active_states:
            self.compteur_hors_seuil = 0
            self.shift_buffer_g.clear()
            self.shift_buffer_d.clear()
            self.prev_gray = gray.copy()
            return self.faute

        ankle_centers = self._valid_ankles(points)
        if ankle_centers is not None:
            center_g_new, center_d_new = ankle_centers

            shift_g_x = abs(center_g_new[0] - self.ref_xg_init) if self.ref_xg_init is not None else 0.0
            shift_g_y = abs(center_g_new[1] - self.ref_yg_init) if self.ref_yg_init is not None else 0.0
            shift_d_x = abs(center_d_new[0] - self.ref_xd_init) if self.ref_xd_init is not None else 0.0
            shift_d_y = abs(center_d_new[1] - self.ref_yd_init) if self.ref_yd_init is not None else 0.0

            shift_g = math.hypot(shift_g_x, shift_g_y * self.vertical_weight)
            shift_d = math.hypot(shift_d_x, shift_d_y * self.vertical_weight)

            self.shift_buffer_g.append(shift_g)
            self.shift_buffer_d.append(shift_d)

            score_g = self._median(self.shift_buffer_g)
            score_d = self._median(self.shift_buffer_d)
            score = max(score_g, score_d)

            if DEBUG_PIEDS and indice_image % DEBUG_PIEDS_EVERY == 0:
                print(
                    f"[PIEDS-POSE] frame={indice_image} etat={etat} "
                    f"shift_g=({shift_g_x:.1f},{shift_g_y:.1f})={shift_g:.1f} "
                    f"shift_d=({shift_d_x:.1f},{shift_d_y:.1f})={shift_d:.1f} "
                    f"score_g={score_g:.1f} score_d={score_d:.1f} "
                    f"score={score:.1f} seuil={self.deplacement_threshold_px:.1f} "
                    f"compteur={self.compteur_hors_seuil}/{self.nb_images_persistance}"
                )

            if score > self.deplacement_threshold_px:
                self.compteur_hors_seuil += 1
            else:
                self.compteur_hors_seuil = max(0, self.compteur_hors_seuil - 2)

            if (not self.faute) and self.compteur_hors_seuil >= self.nb_images_persistance:
                self.faute = True
                if DEBUG_PIEDS:
                    print(f"[PIEDS-FAUTE] frame={indice_image} score={score:.1f} seuil={self.deplacement_threshold_px:.1f} compteur={self.compteur_hors_seuil}/{self.nb_images_persistance}")
                if ajouter_evenement:
                    ajouter_evenement(indice_image, f"FAUTE : déplacement des pieds détecté (score={score:.1f}px > seuil={self.deplacement_threshold_px:.1f}px)")

            self.prev_center_g = center_g_new
            self.prev_center_d = center_d_new
            self.prev_gray = gray.copy()
            return self.faute

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

                shift_g = math.hypot(
                    center_g_new[0] - self.ref_xg_init if self.ref_xg_init is not None else 0.0,
                    center_g_new[1] - self.ref_yg_init if self.ref_yg_init is not None else 0.0,
                )
                shift_d = math.hypot(
                    center_d_new[0] - self.ref_xd_init if self.ref_xd_init is not None else 0.0,
                    center_d_new[1] - self.ref_yd_init if self.ref_yd_init is not None else 0.0,
                )

                # Calculer aussi les composantes individuelles pour le debug
                shift_g_x = abs(center_g_new[0] - self.ref_xg_init) if self.ref_xg_init is not None else 0.0
                shift_g_y = abs(center_g_new[1] - self.ref_yg_init) if self.ref_yg_init is not None else 0.0
                shift_d_x = abs(center_d_new[0] - self.ref_xd_init) if self.ref_xd_init is not None else 0.0
                shift_d_y = abs(center_d_new[1] - self.ref_yd_init) if self.ref_yd_init is not None else 0.0

                # En squat, les keypoints de cheville bougent souvent en Y avec la
                # flexion et la perspective sans vrai deplacement du pied au sol.
                shift_g = math.hypot(shift_g_x, shift_g_y * self.vertical_weight)
                shift_d = math.hypot(shift_d_x, shift_d_y * self.vertical_weight)

                self.shift_buffer_g.append(shift_g)
                self.shift_buffer_d.append(shift_d)

                score_g = self._median(self.shift_buffer_g)
                score_d = self._median(self.shift_buffer_d)
                score = max(score_g, score_d)

                if DEBUG_PIEDS and indice_image % DEBUG_PIEDS_EVERY == 0:
                    print(
                        f"[PIEDS-FLOW] frame={indice_image} etat={etat} "
                        f"disp_g={disp_g:.1f} disp_d={disp_d:.1f} "
                        f"shift_g=({shift_g_x:.1f},{shift_g_y:.1f})={shift_g:.1f} "
                        f"shift_d=({shift_d_x:.1f},{shift_d_y:.1f})={shift_d:.1f} "
                        f"score_g={score_g:.1f} score_d={score_d:.1f} "
                        f"score={score:.1f} seuil={self.deplacement_threshold_px:.1f} "
                        f"compteur={self.compteur_hors_seuil}/{self.nb_images_persistance}"
                    )

                if score > self.deplacement_threshold_px:
                    self.compteur_hors_seuil += 1
                else:
                    self.compteur_hors_seuil = 0

                if (not self.faute) and self.compteur_hors_seuil >= self.nb_images_persistance:
                    self.faute = True
                    if DEBUG_PIEDS:
                        print(f"[PIEDS-FAUTE] frame={indice_image} score={score:.1f} seuil={self.deplacement_threshold_px:.1f} compteur={self.compteur_hors_seuil}/{self.nb_images_persistance}")
                    if ajouter_evenement:
                        ajouter_evenement(indice_image, f"FAUTE : dÃ©placement des pieds dÃ©tectÃ© (score={score:.1f}px > seuil={self.deplacement_threshold_px:.1f}px)")

                self.prev_center_g = center_g_new
                self.prev_center_d = center_d_new

            except Exception as e:
                if DEBUG_PIEDS:
                    print(f"[PIEDS-ERROR] frame={indice_image} {str(e)}")

        self.prev_gray = gray.copy()
        return self.faute
