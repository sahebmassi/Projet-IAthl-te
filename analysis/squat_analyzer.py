"""Analyseur du squat.

Responsabilités principales:
- vue face: détection des phases du squat, profondeur, redescente/dip, pieds;
- vue latérale: suivi disque indépendant de la face, trajectoires descente
  verte et remontée jaune, vitesses moyennes de phase;
- tableau de bord: affiche les repères temporels, vitesses, fautes et journaux.

Principe important de découplage:
la vue face ne doit pas effacer ni piloter la trajectoire latérale. La latérale
détecte ses propres phases depuis le mouvement vertical du disque. Les vitesses
latérales utilisent lateral_fps, pas le FPS de la face.
"""

from collections import deque
from statistics import median
from typing import List, Tuple

import cv2

from .barbell_tracking import (
    DetectionSignal,
    draw_detection_signal,
    phase_barre_depuis_vue_laterale,
    resolve_barbell_target_class,
    tracked_detection_signal,
    tracer_trajectoire,
)
from .base_analyzer import BaseMovementAnalyzer
from .constants import (
    A_MIN_DIP_RATIO_OF_HIPWIDTH,
    BANDE_HAUT_RATIO_LARGEUR_BASSIN,
    BARBELL_CONF_THRES,
    BARBELL_SMOOTH_WINDOW,
    DEBUG_PIEDS,
    DEBUG_PIEDS_EVERY,
    DELAI_MIN_AVANT_FIN_SEC,
    DISK_MODEL_DEFAULT,
    DUREE_HAUT_SEC,
    DUREE_MIN_DESCENTE_A_60_IPS,
    DUREE_VERROUILLAGE_SEC,
    FENETRE_LISSAGE,
    FOOT_CALIBRATION_FRAMES,
    FOOT_DISPLACEMENT_THRESHOLD,
    FOOT_FLOW_ACCUMULATION_WINDOW,
    FOOT_GRACE_PERIOD_FRAMES,
    FOOT_PERSISTANCE_FRAMES,
    FOOT_VERTICAL_WEIGHT,
    K_MIN_DIP_SEC,
    NB_IMAGES_CONSEC_DESCENTE,
    NB_IMAGES_CONSEC_REMONTEE,
    NB_IMAGES_CONSEC_BARRE_PHASE,
    SEUIL_ANGLE_GENOU_PROFONDEUR,
    SEUIL_STABILITE_RATIO_LARGEUR_BASSIN,
    SEUIL_VERROUILLAGE_GENOU,
    SEUIL_VITESSE_RATIO_LARGEUR_BASSIN,
    SQUAT_FOOT_DISPLACEMENT_RATIO_HIPWIDTH,
    X_IGNORE_AFTER_ASCENT_SEC,
)
from .foot_detector import DetecteurPiedsRobuste
from .geometry import angle_moyen_genoux, y_bassin_et_largeur, y_moyenne_genoux
from .pose import dessiner_squelette


class SquatAnalyzer(BaseMovementAnalyzer):
    """Analyseur complet du squat.

    La vue face sert à juger le mouvement du corps: début/fin de phase,
    profondeur, redescente et pieds. La vue latérale sert à suivre le disque ou
    la barre et à calculer des vitesses. Les deux vues restent découplées pour
    éviter qu'une vidéo désynchronisée efface les résultats de l'autre.
    """

    dashboard_title = "TABLEAU DE BORD - SQUAT"

    def __init__(self, **kwargs):
        """Initialise tout l'état nécessaire à l'analyse du squat.

        Le constructeur prépare:
        - les seuils exprimés en nombre d'images à partir du FPS source;
        - les buffers de lissage du bassin, des genoux et du disque;
        - les classes YOLO à suivre;
        - les variables de machine d'état de la vue face;
        - les variables de phase, trajectoire et vitesse de la vue latérale;
        - le détecteur robuste de déplacement des pieds.

        Beaucoup d'attributs sont stockés sur self car process_frame() est appelé
        image par image par VideoWorker et doit conserver la mémoire des images
        précédentes.
        """

        super().__init__(require_barbell_model=True, **kwargs)

        self.duree_min_descente_images = max(
            2, int(round(DUREE_MIN_DESCENTE_A_60_IPS * (self.fps / 60.0)))
        )
        self.verrouillage_images = max(
            2, int(round(DUREE_VERROUILLAGE_SEC * self.fps))
        )
        self.maintien_haut_images = max(2, int(round(DUREE_HAUT_SEC * self.fps)))
        self.delai_min_avant_fin_images = max(
            0, int(round(DELAI_MIN_AVANT_FIN_SEC * self.fps))
        )
        self.k_min_dip_frames = max(2, int(round(K_MIN_DIP_SEC * self.fps)))
        self.x_ignore_frames = max(0, int(round(X_IGNORE_AFTER_ASCENT_SEC * self.fps)))

        self.buffer_y_bassin = deque(maxlen=FENETRE_LISSAGE)
        self.buffer_largeur_bassin = deque(maxlen=FENETRE_LISSAGE)
        self.buffer_angle_genou = deque(maxlen=FENETRE_LISSAGE)
        self.buffer_position_haute = deque(maxlen=int(max(10, round(1.0 * self.fps))))
        self.hist_barbell = deque(maxlen=BARBELL_SMOOTH_WINDOW)
        self.traj_descente = []
        self.traj_remontee = []
        self.barbell_target_class = resolve_barbell_target_class(
            self.barbell_model_path,
            self.barbell_model.names,
            preferred_kind="auto",
        )

        self.disk_model = None
        self.disk_positions = []
        self.disk_positions_by_frame = {}
        self.cm_per_pixel = None
        self.speed_descente = None
        self.speed_remontee = None
        self.speed_descente_details = None
        self.speed_remontee_details = None
        self.disk_descente = []
        self.disk_remontee = []
        self.disk_descente_frames = []
        self.disk_remontee_frames = []
        self.lateral_disk_frozen = False
        if len(self.video_paths) > 1:
            from ultralytics import YOLO
            self.disk_model = YOLO(DISK_MODEL_DEFAULT)

        self.etat = "attente"
        self.indice_image = -1
        self.suite_descente = 0
        self.suite_remontee = 0
        self.y_bassin_lisse_avant = None
        self.image_debut_descente = None
        self.image_debut_remontee = None
        self.image_fin_remontee = None
        self.y_position_haute = None
        self.phase_barre_laterale = None
        self.phase_barre_laterale_precedente = None
        self.y_barre_precedent = None
        self.compteur_descente_barre = 0
        self.compteur_remontee_barre = 0
        self.image_debut_descente_barre = None
        self.image_debut_remontee_barre = None
        self.image_fin_remontee_barre = None
        self.y_disk_precedent = None
        self.phase_disk_laterale = None
        self.phase_disk_laterale_precedente = None
        self.compteur_descente_disk = 0
        self.compteur_remontee_disk = 0
        self.image_debut_descente_disk = None
        self.image_debut_remontee_disk = None
        self.image_fin_remontee_disk = None
        self.image_point_bas = None
        self.angle_genou_point_bas = None
        self.y_genoux_point_bas = None
        self.faute_descente_insuffisante = False
        self.hanches_sous_genoux = None
        self.angle_genou_ok = None
        self.angle_genou_min_observe = None
        self.dip_detected = False
        self.dip_start_frame = None
        self.dip_amp_px = None
        self.dip_streak = 0
        self.dip_candidate_start = None
        self.dip_base_y = None
        self.dip_peak_y = None
        self.suite_verrouillage = 0
        self.suite_haut = 0
        self.detecteur_pieds = DetecteurPiedsRobuste(
            nb_images_calibration=FOOT_CALIBRATION_FRAMES,
            flow_accumulation_window=FOOT_FLOW_ACCUMULATION_WINDOW,
            deplacement_threshold_px=FOOT_DISPLACEMENT_THRESHOLD,
            nb_images_persistance=FOOT_PERSISTANCE_FRAMES,
            grace_period_frames=FOOT_GRACE_PERIOD_FRAMES,
            vertical_weight=FOOT_VERTICAL_WEIGHT,
            ignorer_apres_fin=True,
            debug=DEBUG_PIEDS,
        )
        self.dynamic_foot_threshold_set = False

        self.last_largeur_bassin = None
        self.last_vitesse = None
        self.last_angle_genou = None
        self.last_foot_fault = False
        self.add_event(-1, "Analyse squat demarree")
        if self.barbell_target_class is not None:
            self.add_event(
                -1,
                f"Suivi barre actif avec la classe '{self.barbell_target_class}'",
            )

    def process_frame(self, frames: List, source_frame_index: int):
        """Analyse un paquet de frames pour le squat.

        La vue face met à jour l'état du squat et les fautes liées au corps. La
        vue latérale suit le disque indépendamment et conserve ses propres
        repères de phase. Ainsi, une vue face non synchronisée ne peut pas
        effacer ni forcer la trajectoire latérale.
        """

        self.indice_image = source_frame_index
        vues_annotees = [f.copy() for f in frames if f is not None]
        video_face = vues_annotees[0] if vues_annotees else None
        video_barre = vues_annotees[1] if len(vues_annotees) > 1 else None

        points = None
        if video_face is not None:
            points = self.predict_pose_points(video_face)
            if points is not None:
                dessiner_squelette(video_face, points)

            y_bassin, largeur_bassin = y_bassin_et_largeur(points)
            if y_bassin is not None:
                self.buffer_y_bassin.append(y_bassin)
                self.buffer_position_haute.append(y_bassin)
                if largeur_bassin is not None:
                    self.buffer_largeur_bassin.append(largeur_bassin)

            angle_genou = angle_moyen_genoux(points)
            if angle_genou is not None:
                self.buffer_angle_genou.append(angle_genou)

        y_bassin_lisse = None
        largeur_bassin_lisse = None
        angle_genou_lisse = None
        vitesse = None

        if len(self.buffer_y_bassin) >= max(3, FENETRE_LISSAGE // 2):
            y_bassin_lisse = float(median(self.buffer_y_bassin))
            largeur_bassin_lisse = (
                float(median(self.buffer_largeur_bassin))
                if self.buffer_largeur_bassin
                else 200.0
            )
            angle_genou_lisse = (
                float(median(self.buffer_angle_genou))
                if self.buffer_angle_genou
                else None
            )

            if angle_genou_lisse is not None:
                if (
                    self.angle_genou_min_observe is None
                    or angle_genou_lisse < self.angle_genou_min_observe
                ):
                    self.angle_genou_min_observe = angle_genou_lisse

            seuil_vitesse_px = SEUIL_VITESSE_RATIO_LARGEUR_BASSIN * largeur_bassin_lisse
            seuil_stabilite_px = (
                SEUIL_STABILITE_RATIO_LARGEUR_BASSIN * largeur_bassin_lisse
            )
            bande_haut_px = BANDE_HAUT_RATIO_LARGEUR_BASSIN * largeur_bassin_lisse
            a_min_dip_px = A_MIN_DIP_RATIO_OF_HIPWIDTH * largeur_bassin_lisse

            if self.y_bassin_lisse_avant is not None:
                vitesse = y_bassin_lisse - self.y_bassin_lisse_avant
                self.suite_descente = (
                    self.suite_descente + 1 if vitesse > seuil_vitesse_px else 0
                )
                self.suite_remontee = (
                    self.suite_remontee + 1 if vitesse < -seuil_vitesse_px else 0
                )

                if self.etat == "attente":
                    if self.suite_descente >= NB_IMAGES_CONSEC_DESCENTE:
                        self.etat = "descente"
                        self.image_debut_descente = (
                            self.indice_image - NB_IMAGES_CONSEC_DESCENTE + 1
                        )
                        self.y_position_haute = (
                            float(median(self.buffer_position_haute))
                            if self.buffer_position_haute
                            else self.y_bassin_lisse_avant
                        )
                        self.y_bassin_max = y_bassin_lisse
                        self.image_point_bas = self.indice_image
                        self.angle_genou_point_bas = angle_genou_lisse
                        self.y_genoux_point_bas = (
                            y_moyenne_genoux(points) if points is not None else None
                        )
                        if (
                            largeur_bassin_lisse is not None
                            and not self.dynamic_foot_threshold_set
                        ):
                            self.detecteur_pieds.deplacement_threshold_px = max(
                                FOOT_DISPLACEMENT_THRESHOLD,
                                SQUAT_FOOT_DISPLACEMENT_RATIO_HIPWIDTH
                                * largeur_bassin_lisse,
                            )
                            self.dynamic_foot_threshold_set = True
                        self.add_event(
                            self.indice_image,
                            f"Debut de descente (frame {self.image_debut_descente})",
                        )

                elif self.etat == "descente":
                    if self.y_bassin_max is None or y_bassin_lisse > self.y_bassin_max:
                        self.y_bassin_max = y_bassin_lisse
                        self.image_point_bas = self.indice_image
                        self.angle_genou_point_bas = angle_genou_lisse
                        self.y_genoux_point_bas = (
                            y_moyenne_genoux(points) if points is not None else None
                        )

                    if (
                        self.image_debut_descente is not None
                        and (self.indice_image - self.image_debut_descente)
                        >= self.duree_min_descente_images
                        and self.suite_remontee >= NB_IMAGES_CONSEC_REMONTEE
                    ):
                        self.etat = "remontee"
                        self.image_debut_remontee = (
                            self.indice_image - NB_IMAGES_CONSEC_REMONTEE + 1
                        )
                        self.add_event(
                            self.indice_image,
                            f"Debut de remontee (frame {self.image_debut_remontee})",
                        )

                        if self.y_bassin_max is not None:
                            if self.y_genoux_point_bas is not None:
                                self.hanches_sous_genoux = (
                                    self.y_bassin_max > self.y_genoux_point_bas
                                )
                            else:
                                self.hanches_sous_genoux = None

                            if self.angle_genou_point_bas is not None:
                                self.angle_genou_ok = (
                                    self.angle_genou_point_bas
                                    < SEUIL_ANGLE_GENOU_PROFONDEUR
                                )
                            else:
                                self.angle_genou_ok = None

                            profondeur_ok = (
                                self.hanches_sous_genoux is True
                            ) or (self.angle_genou_ok is True)
                            self.faute_descente_insuffisante = not profondeur_ok
                            self.add_event(
                                self.indice_image,
                                "FAUTE : descente non suffisante"
                                if self.faute_descente_insuffisante
                                else "OK : descente suffisante",
                            )

                        self.dip_streak = 0
                        self.dip_candidate_start = None
                        self.dip_base_y = None
                        self.dip_peak_y = None
                        self.suite_verrouillage = 0
                        self.suite_haut = 0

                elif self.etat == "remontee":
                    if (
                        self.image_debut_remontee is not None
                        and (self.indice_image - self.image_debut_remontee)
                        >= self.delai_min_avant_fin_images
                    ):
                        stable = abs(vitesse) < seuil_stabilite_px

                        if (
                            angle_genou_lisse is not None
                            and angle_genou_lisse >= SEUIL_VERROUILLAGE_GENOU
                            and stable
                        ):
                            self.suite_verrouillage += 1
                        else:
                            self.suite_verrouillage = 0

                        if (
                            self.y_position_haute is not None
                            and abs(y_bassin_lisse - self.y_position_haute) <= bande_haut_px
                            and stable
                        ):
                            self.suite_haut += 1
                        else:
                            self.suite_haut = 0

                        if (
                            self.suite_verrouillage >= self.verrouillage_images
                            or self.suite_haut >= self.maintien_haut_images
                        ):
                            self.etat = "termine"
                            self.image_fin_remontee = self.indice_image
                            raison = (
                                "genoux verrouilles"
                                if self.suite_verrouillage >= self.verrouillage_images
                                else "retour en haut"
                            )
                            self.add_event(
                                self.indice_image,
                                f"Fin de remontee ({raison}, frame {self.image_fin_remontee})",
                            )

                    if (
                        not self.dip_detected
                        and self.image_debut_remontee is not None
                        and (self.indice_image - self.image_debut_remontee)
                        >= self.x_ignore_frames
                    ):
                        if vitesse > seuil_vitesse_px:
                            if self.dip_candidate_start is None:
                                self.dip_candidate_start = self.indice_image
                                self.dip_base_y = self.y_bassin_lisse_avant
                                self.dip_peak_y = y_bassin_lisse
                            self.dip_streak += 1
                            self.dip_peak_y = max(self.dip_peak_y, y_bassin_lisse)
                        elif self.dip_candidate_start is not None:
                            amplitude = (
                                self.dip_peak_y - self.dip_base_y
                                if self.dip_peak_y is not None and self.dip_base_y is not None
                                else 0.0
                            )
                            if (
                                self.dip_streak >= self.k_min_dip_frames
                                and amplitude >= a_min_dip_px
                            ):
                                self.dip_detected = True
                                self.dip_start_frame = self.dip_candidate_start
                                self.dip_amp_px = amplitude
                                self.add_event(
                                    self.indice_image,
                                    f"FAUTE : redescente detectee (frame {self.dip_start_frame}, amplitude {amplitude:.1f}px)",
                                )
                            self.dip_streak = 0
                            self.dip_candidate_start = None
                            self.dip_base_y = None
                            self.dip_peak_y = None

            self.y_bassin_lisse_avant = y_bassin_lisse

        if video_barre is not None:
            resultats_barbell = self.barbell_model.predict(
                source=video_barre,
                conf=BARBELL_CONF_THRES,
                verbose=False,
            )
            result_barbell = resultats_barbell[0] if resultats_barbell else None
            barbell_signal = tracked_detection_signal(
                result_barbell,
                self.barbell_model.names,
                self.barbell_target_class,
                self.hist_barbell,
                "barre",
            )
            if barbell_signal.center is not None:
                draw_detection_signal(
                    video_barre,
                    barbell_signal,
                    (0, 255, 0),
                    (0, 0, 255),
                    "barre",
                )

            if barbell_signal.center is not None:
                (
                    self.phase_barre_laterale,
                    self.compteur_descente_barre,
                    self.compteur_remontee_barre,
                ) = phase_barre_depuis_vue_laterale(
                    barbell_signal.center[1],
                    self.y_barre_precedent,
                    self.phase_barre_laterale,
                    self.compteur_descente_barre,
                    self.compteur_remontee_barre,
                )
                self.y_barre_precedent = barbell_signal.center[1]
                if (
                    self.phase_barre_laterale == "descente"
                    and self.phase_barre_laterale_precedente != "descente"
                    and self.image_debut_descente_barre is None
                ):
                    self.image_debut_descente_barre = max(
                        0, self.indice_image - NB_IMAGES_CONSEC_BARRE_PHASE + 1
                    )
                if (
                    self.phase_barre_laterale == "remontee"
                    and self.phase_barre_laterale_precedente != "remontee"
                    and self.image_debut_remontee_barre is None
                ):
                    self.image_debut_remontee_barre = max(
                        0, self.indice_image - NB_IMAGES_CONSEC_BARRE_PHASE + 1
                    )
                if (
                    self.phase_barre_laterale_precedente == "remontee"
                    and self.phase_barre_laterale == "descente"
                    and self.image_fin_remontee_barre is None
                ):
                    self.image_fin_remontee_barre = self.indice_image
                self.phase_barre_laterale_precedente = self.phase_barre_laterale
                phase_barre = self.phase_disk_laterale
            else:
                phase_barre = self.phase_disk_laterale
        else:
            barbell_signal = DetectionSignal(source="barre")
            phase_barre = self.phase_disk_laterale

        if self.disk_model is not None and video_barre is not None:
            result_disk = self.disk_model.predict(source=video_barre, conf=BARBELL_CONF_THRES, verbose=False)
            if result_disk:
                result = result_disk[0]
                if result.boxes:
                    box = result.boxes[0]
                    bbox = box.xyxy[0].cpu().numpy()
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    diameter_px = bbox[2] - bbox[0]
                    if self.cm_per_pixel is None and diameter_px > 0:
                        self.cm_per_pixel = 45.0 / diameter_px
                    disk_point = (int(center[0]), int(center[1]))
                    self.disk_positions.append(disk_point)
                    self.disk_positions_by_frame[self.indice_image] = disk_point
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(video_barre, (x1, y1), (x2, y2), (0, 180, 255), 2)
                    cv2.putText(
                        video_barre,
                        "DISQUE",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 180, 255),
                        2,
                    )
                    cv2.circle(video_barre, disk_point, 5, (255, 0, 0), -1)

                    if self.y_disk_precedent is not None:
                        (
                            self.phase_disk_laterale,
                            self.compteur_descente_disk,
                            self.compteur_remontee_disk,
                        ) = phase_barre_depuis_vue_laterale(
                            center[1],
                            self.y_disk_precedent,
                            self.phase_disk_laterale,
                            self.compteur_descente_disk,
                            self.compteur_remontee_disk,
                        )
                        if (
                            self.phase_disk_laterale == "descente"
                            and self.phase_disk_laterale_precedente != "descente"
                            and not self.lateral_disk_frozen
                        ):
                            if (
                                self.phase_disk_laterale_precedente == "remontee"
                                and self.disk_remontee_frames
                            ):
                                self.image_fin_remontee_disk = self.disk_remontee_frames[-1][0]
                                self.lateral_disk_frozen = True
                            elif self.image_debut_descente_disk is None:
                                self._reset_lateral_disk_phase()
                                self.image_debut_descente_disk = max(
                                    0,
                                    self.indice_image - NB_IMAGES_CONSEC_BARRE_PHASE + 1,
                                )
                        if (
                            self.phase_disk_laterale == "remontee"
                            and self.phase_disk_laterale_precedente != "remontee"
                            and self.image_debut_remontee_disk is None
                        ):
                            self.image_debut_remontee_disk = max(
                                0, self.indice_image - NB_IMAGES_CONSEC_BARRE_PHASE + 1
                            )
                        self.phase_disk_laterale_precedente = self.phase_disk_laterale
                    self.y_disk_precedent = center[1]

        phase_barre = self.phase_disk_laterale

        if phase_barre == "descente" and not self.lateral_disk_frozen:
            if hasattr(self, 'disk_descente') and self.disk_positions:
                point = self.disk_positions[-1]
                self.disk_descente.append(point)
                self.disk_descente_frames.append((self.indice_image, point))
        elif phase_barre == "remontee" and not self.lateral_disk_frozen:
            if hasattr(self, 'disk_remontee') and self.disk_positions:
                point = self.disk_positions[-1]
                self.disk_remontee.append(point)
                self.disk_remontee_frames.append((self.indice_image, point))

        self._update_phase_average_speeds()

        if video_barre is not None and (self.disk_descente or self.disk_remontee):
            tracer_trajectoire(video_barre, self.disk_descente, (0, 255, 0))
            tracer_trajectoire(video_barre, self.disk_remontee, (0, 220, 255))

        if video_face is not None:
            faute_pieds = self.detecteur_pieds.update(
                frame=video_face,
                etat=self.etat,
                points=points,
                indice_image=self.indice_image,
                ajouter_evenement=self.add_event,
                active_states={"descente", "remontee"},
            )
        else:
            faute_pieds = False
        self.last_foot_fault = faute_pieds
        self.last_largeur_bassin = largeur_bassin_lisse
        self.last_vitesse = vitesse
        self.last_angle_genou = angle_genou_lisse

        if DEBUG_PIEDS and self.indice_image % DEBUG_PIEDS_EVERY == 0:
            yb = f"{y_bassin_lisse:.1f}" if y_bassin_lisse is not None else "-"
            vit = f"{vitesse:+.2f}" if vitesse is not None else "-"
            lb = f"{largeur_bassin_lisse:.1f}" if largeur_bassin_lisse is not None else "-"
            ag = f"{angle_genou_lisse:.1f}" if angle_genou_lisse is not None else "-"
            print(
                f"[SQUAT] frame={self.indice_image} etat={self.etat} "
                f"y_bassin={yb} vitesse={vit} largeur_bassin={lb} angle_genou={ag} "
                f"faute_pieds={faute_pieds}"
            )

        t_sec = self.indice_image / self.fps
        verdict_global, sections = self._build_dashboard_sections(
            y_bassin_lisse,
            largeur_bassin_lisse,
            vitesse,
            angle_genou_lisse,
        )
        if video_face is not None:
            self._draw_overlay_header(
                video_face,
                "Squat",
                f"Etat: {self.etat} | Verdict: {verdict_global}",
                (80, 220, 255),
            )
        return self.compose_output(
            vues_annotees,
            ips_txt=self.format_fps_text(),
            indice_image=self.indice_image,
            t_sec=t_sec,
            etat=self.etat,
            verdict_global=verdict_global,
            sections=sections,
        )

    def _reset_lateral_disk_phase(self) -> None:
        """Réinitialise uniquement les données de phase du disque latéral.

        Cette méthode ne doit pas être appelée depuis un événement de la vue
        face. Elle est réservée aux transitions détectées par la vue latérale
        elle-même, lorsqu'une nouvelle descente latérale commence.
        """

        self.disk_descente.clear()
        self.disk_remontee.clear()
        self.disk_descente_frames.clear()
        self.disk_remontee_frames.clear()
        self.speed_descente = None
        self.speed_remontee = None
        self.speed_descente_details = None
        self.speed_remontee_details = None
        self.image_debut_descente_disk = None
        self.image_debut_remontee_disk = None
        self.image_fin_remontee_disk = None
        self.lateral_disk_frozen = False

    def _calculate_average_speed(
        self,
        phase_points: List[Tuple[int, Tuple[int, int]]],
    ):
        """Calcule la vitesse moyenne d'une phase latérale.

        phase_points contient des couples (frame_source, (x, y)). La méthode
        prend le point le plus haut et le point le plus bas de la phase,
        convertit la distance verticale en centimètres avec cm_per_pixel, puis
        divise par le temps calculé avec lateral_fps.

        Retourne un dictionnaire de détails pour le tableau de bord, ou None si
        l'échelle ou les points disponibles sont insuffisants.
        """

        if self.cm_per_pixel is None or len(phase_points) < 2:
            return None

        top_frame, top_point = min(phase_points, key=lambda item: item[1][1])
        bottom_frame, bottom_point = max(phase_points, key=lambda item: item[1][1])

        delta_px = abs(bottom_point[1] - top_point[1])
        duration_s = abs(bottom_frame - top_frame) / self.lateral_fps
        if duration_s <= 0:
            return None

        distance_cm = delta_px * self.cm_per_pixel
        speed_cm_s = distance_cm / duration_s
        return {
            "speed_cm_s": speed_cm_s,
            "distance_cm": distance_cm,
            "delta_px": delta_px,
            "duration_s": duration_s,
            "top_frame": top_frame,
            "bottom_frame": bottom_frame,
            "top_y": top_point[1],
            "bottom_y": bottom_point[1],
        }

    def _update_phase_average_speeds(self) -> None:
        """Calcule et fige les vitesses moyennes latérales.

        La vitesse de descente est calculée quand la vue latérale a détecté le
        début de remontée. La vitesse de remontée est calculée quand la vue
        latérale a détecté la fin de remontée. Une fois calculées, les valeurs
        restent affichées et ne sont pas recalculées à chaque frame.
        """

        if self.speed_descente is None and self.image_debut_remontee_disk is not None:
            descente = self._calculate_average_speed(self.disk_descente_frames)
            if descente is not None:
                self.speed_descente_details = descente
                self.speed_descente = descente["speed_cm_s"]
                self.add_event(
                    max(descente["top_frame"], descente["bottom_frame"]),
                    f"Vitesse moyenne descente: {self.speed_descente:.2f} cm/s",
                )

        if self.speed_remontee is None and self.image_fin_remontee_disk is not None:
            remontee = self._calculate_average_speed(self.disk_remontee_frames)
            if remontee is not None:
                self.speed_remontee_details = remontee
                self.speed_remontee = remontee["speed_cm_s"]
                self.add_event(
                    max(remontee["top_frame"], remontee["bottom_frame"]),
                    f"Vitesse moyenne remontee: {self.speed_remontee:.2f} cm/s",
                )

    def _build_dashboard_sections(
        self,
        y_bassin_lisse,
        largeur_bassin_lisse,
        vitesse,
        angle_genou_lisse,
    ):
        """Construit les sections du tableau de bord squat.

        Cette méthode rassemble seulement l'état déjà calculé: repères
        temporels, données lissées, fautes, vitesse disque et verdict. Elle ne
        doit pas modifier la logique d'analyse.
        """

        verdict_profondeur = "-"
        if self.etat in ("remontee", "termine") and self.image_debut_remontee is not None:
            verdict_profondeur = "FAUTE" if self.faute_descente_insuffisante else "OK"

        fautes = []
        if self.faute_descente_insuffisante:
            fautes.append("profondeur")
        if self.dip_detected:
            fautes.append("redescente")
        if self.last_foot_fault:
            fautes.append("pieds")

        if not fautes and self.etat == "termine":
            verdict_global = "VALIDE"
        elif fautes:
            verdict_global = "FAUTE : " + ", ".join(fautes)
        else:
            verdict_global = "EN COURS"

        sections = [
            (
                "REPÈRES TEMPORELS (VUE FACE)",
                [
                    f"Début descente : {self.image_debut_descente if self.image_debut_descente is not None else '—'}",
                    f"Début remontée : {self.image_debut_remontee if self.image_debut_remontee is not None else '—'}",
                    f"Fin remontée   : {self.image_fin_remontee if self.image_fin_remontee is not None else '—'}",
                ],
            ),
            (
                "REPÈRES TEMPORELS (VUE LATÉRALE)",
                [
                    f"État disque latéral : {self.phase_disk_laterale if self.phase_disk_laterale is not None else '—'}",
                    f"Début descente : {self.image_debut_descente_disk if self.image_debut_descente_disk is not None else '—'}",
                    f"Début remontée : {self.image_debut_remontee_disk if self.image_debut_remontee_disk is not None else '—'}",
                    f"Fin remontée   : {self.image_fin_remontee_disk if self.image_fin_remontee_disk is not None else '—'}",
                ],
            ),
            (
                "DONNÉES LISSÉES",
                [
                    f"Y bassin (px)        : {f'{y_bassin_lisse:.1f}' if y_bassin_lisse is not None else '—'}",
                    f"Largeur bassin (px)  : {f'{largeur_bassin_lisse:.1f}px' if largeur_bassin_lisse is not None else '—'}",
                    f"Vitesse bassin (px/f): {f'{vitesse:+.2f}px' if vitesse is not None else '—'}",
                    f"Angle genou (°)      : {f'{angle_genou_lisse:.1f}°' if angle_genou_lisse is not None else '—'}",
                    f"Angle genou min obs. : {f'{self.angle_genou_min_observe:.1f}°' if self.angle_genou_min_observe is not None else '—'}",
                ],
            ),
            (
                "ANALYSE REDESCENTE (DIP)",
                [
                    f"Dip détecté ?        : {'Détecté' if self.dip_detected else 'Non détecté'}",
                    f"Frame début dip      : {self.dip_start_frame if self.dip_start_frame is not None else '—'}",
                    f"Amplitude dip (px)   : {self.dip_amp_px:.1f}" if self.dip_amp_px is not None else "Amplitude dip (px)   : —",
                ],
            ),
            (
                "ANALYSE PROFONDEUR",
                [
                    f"Image point bas      : {self.image_point_bas if self.image_point_bas is not None else '—'}",
                    f"Angle genou au point bas : {f'{self.angle_genou_point_bas:.1f}°' if self.angle_genou_point_bas is not None else '—'}",
                    f"Hanches sous genoux  : {self.hanches_sous_genoux if self.hanches_sous_genoux is not None else '—'}",
                    f"Angle < {SEUIL_ANGLE_GENOU_PROFONDEUR:.0f}° : {self.angle_genou_ok if self.angle_genou_ok is not None else '—'}",
                    f"Verdict profondeur   : {verdict_profondeur}",
                ],
            ),
            (
                "ANALYSE PIEDS",
                [
                    f"Calibration OK       : {'Oui' if self.detecteur_pieds.calibration_faite else 'Non (en cours)'}",
                    f"Seuil dynamique (px) : {self.detecteur_pieds.deplacement_threshold_px:.1f}",
                    f"État                 : {'FAUTE' if self.last_foot_fault else 'OK'}",
                    f"Compteur hors seuil  : {self.detecteur_pieds.compteur_hors_seuil}/{self.detecteur_pieds.nb_images_persistance}",
                ],
            ),
        ]

        if len(self.video_paths) > 1:
            if self.cm_per_pixel is not None:
                suivi_disque = "OK"
            elif self.disk_model is not None:
                suivi_disque = "En attente de détection"
            else:
                suivi_disque = "Modèle disque indisponible"

            vitesse_lignes = [
                f"Suivi disque     : {suivi_disque}",
                f"Points détectés  : {len(self.disk_positions)}",
                f"Échelle          : {self.cm_per_pixel:.4f} cm/px" if self.cm_per_pixel is not None else "Échelle          : —",
                f"FPS latéral      : {self.lateral_fps:.2f}",
                f"Vitesse descente moyenne : {self.speed_descente:.2f} cm/s" if self.speed_descente is not None else "Vitesse descente moyenne : —",
            ]
            if self.speed_descente_details is not None:
                details = self.speed_descente_details
                vitesse_lignes.extend(
                    [
                        f"Descente frames  : haut {details['top_frame']} | bas {details['bottom_frame']}",
                        f"Descente Y disque: haut {details['top_y']} px | bas {details['bottom_y']} px",
                        f"Descente distance: {details['delta_px']:.1f}px = {details['distance_cm']:.2f} cm",
                        f"Descente temps   : {details['duration_s']:.3f} s",
                    ]
                )

            vitesse_lignes.append(
                f"Vitesse remontée moyenne : {self.speed_remontee:.2f} cm/s" if self.speed_remontee is not None else "Vitesse remontée moyenne : —"
            )
            if self.speed_remontee_details is not None:
                details = self.speed_remontee_details
                vitesse_lignes.extend(
                    [
                        f"Remontée frames  : bas {details['bottom_frame']} | haut {details['top_frame']}",
                        f"Remontée Y disque: bas {details['bottom_y']} px | haut {details['top_y']} px",
                        f"Remontée distance: {details['delta_px']:.1f}px = {details['distance_cm']:.2f} cm",
                        f"Remontée temps   : {details['duration_s']:.3f} s",
                    ]
                )

            sections.insert(
                3,
                (
                    "VITESSE DISQUE",
                    vitesse_lignes,
                ),
            )

        return verdict_global, sections
