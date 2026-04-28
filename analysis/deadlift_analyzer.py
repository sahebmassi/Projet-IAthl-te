"""Analyseur du deadlift.

Le deadlift partage les mêmes conventions que le squat, avec une seule phase
latérale utile: la remontée. La vue latérale suit le disque, détecte son début et
sa fin de remontée, gèle la trajectoire une fois terminée et calcule la vitesse
moyenne bas -> haut avec lateral_fps.

La vue face/corps reste responsable du signal corps, de la redescente visible de
face et de l'analyse des pieds. Le verdict actuel est volontairement limité à
deux fautes: redescente et pieds. Les genoux restent affichés pour diagnostic,
mais ne refusent pas encore l'essai.
"""

import os
from collections import deque
from statistics import median
from typing import List, Optional, Tuple

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
    BARBELL_CONF_THRES,
    BARBELL_SMOOTH_WINDOW,
    DEADLIFT_FOOT_DISPLACEMENT_RATIO_HIPWIDTH,
    DEADLIFT_FACE_BAR_MODEL_DEFAULT,
    DEADLIFT_LOCKOUT_KNEE_ANGLE,
    DEADLIFT_REDESCENT_CONSEC_FRAMES,
    DEADLIFT_REDESCENT_MIN_PX,
    DEADLIFT_REDESCENT_RATIO_HIPWIDTH,
    DEADLIFT_SETUP_FRAMES,
    DEADLIFT_START_CONSEC_FRAMES,
    DEADLIFT_START_DISPLACEMENT_MIN_PX,
    DEADLIFT_START_DISPLACEMENT_RATIO_HIPWIDTH,
    DEADLIFT_START_VELOCITY_MIN_PX,
    DEADLIFT_START_VELOCITY_RATIO_HIPWIDTH,
    DEADLIFT_TOP_CANDIDATE_FRAMES,
    DEADLIFT_TOP_HOLD_FRAMES,
    DEADLIFT_TOP_MIN_ASCENT_MIN_PX,
    DEADLIFT_TOP_MIN_ASCENT_RATIO_HIPWIDTH,
    DEADLIFT_TOP_PLATEAU_MIN_PX,
    DEADLIFT_TOP_PLATEAU_RATIO_HIPWIDTH,
    DEADLIFT_TOP_STABILITY_MIN_PX,
    DEADLIFT_TOP_STABILITY_RATIO_HIPWIDTH,
    DEBUG_PIEDS,
    DEBUG_PIEDS_EVERY,
    DISK_MODEL_DEFAULT,
    FENETRE_LISSAGE,
    FOOT_CALIBRATION_FRAMES,
    FOOT_DISPLACEMENT_THRESHOLD,
    FOOT_FLOW_ACCUMULATION_WINDOW,
    FOOT_GRACE_PERIOD_FRAMES,
    FOOT_PERSISTANCE_FRAMES,
    FOOT_VERTICAL_WEIGHT,
    NB_IMAGES_CONSEC_BARRE_PHASE,
)
from .foot_detector import DetecteurPiedsRobuste
from .geometry import angles_genoux, angle_genou_min_visible, y_bassin_et_largeur, y_epaules_et_largeur
from .pose import choisir_personne_principale, dessiner_squelette, dessiner_squelette_lateral


class DeadliftAnalyzer(BaseMovementAnalyzer):
    """Analyseur complet du soulevé de terre.

    La vue face/corps détecte le départ, la redescente visible, les pieds et les
    informations de genoux. La vue latérale suit le disque pour mesurer la phase
    de remontée et calculer sa vitesse. Le verdict actuel reste volontairement
    limité aux fautes de redescente et de pieds.
    """

    dashboard_title = "TABLEAU DE BORD - DEADLIFT"

    def __init__(self, **kwargs):
        """Initialise l'état du deadlift.

        On prépare les buffers de lissage du corps, les buffers du disque
        latéral, les repères de phase, les variables de faute, le modèle barre
        face optionnel et le détecteur de pieds. Comme pour le squat, l'analyse
        est incrémentale: chaque frame dépend de l'historique des frames
        précédentes.
        """

        super().__init__(require_barbell_model=False, require_disk_model=False, **kwargs)

        self.face_bar_model_path = (
            DEADLIFT_FACE_BAR_MODEL_DEFAULT
            if os.path.exists(DEADLIFT_FACE_BAR_MODEL_DEFAULT)
            else None
        )
        self.face_bar_model = self._load_model(
            self.face_bar_model_path,
            "Modele barre face deadlift",
            required=False,
        )
        self.bar_tracking_enabled = len(self.video_paths) > 1 and self.disk_model is not None
        self.disk_target_class = (
            resolve_barbell_target_class(
                self.disk_model_path,
                self.disk_model.names,
                preferred_kind="disk",
            )
            if self.bar_tracking_enabled and self.disk_model is not None
            else None
        )

        self.body_y_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.body_setup_buffer = deque(maxlen=DEADLIFT_SETUP_FRAMES)
        self.body_reference_y = None
        self.last_body_y = None
        self.last_body_speed = None

        self.disk_history = deque(maxlen=BARBELL_SMOOTH_WINDOW)
        self.bar_y_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.bar_x_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.bar_setup_buffer = deque(maxlen=DEADLIFT_SETUP_FRAMES)
        self.bar_reference_y = None
        self.last_bar_y = None
        self.last_bar_speed = None
        self.last_disk_signal = DetectionSignal(source="disque")
        self.last_fused_signal = DetectionSignal(source="disque")
        self.last_bar_source = "disque"
        self.phase_disk_laterale = None
        self.phase_disk_laterale_precedente = None
        self.compteur_descente_disk = 0
        self.compteur_remontee_disk = 0
        self.image_debut_remontee_disk = None
        self.image_fin_remontee_disk = None
        self.lateral_disk_frozen = False
        self.lateral_top_streak = 0
        self.last_face_bar_points = None
        self.last_face_bar_center = None
        self.last_face_bar_detected = False

        self.hip_width_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.knee_left_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.knee_right_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.final_lockout_window = deque(maxlen=DEADLIFT_TOP_HOLD_FRAMES)
        self.bar_path_points = []

        self.disk_positions = []
        self.cm_per_pixel = None
        self.speed_remontee = None
        self.speed_remontee_details = None
        self.disk_descente = []
        self.disk_remontee = []
        self.disk_remontee_frames = []

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

        self.indice_image = -1
        self.etat = "attente"

        self.primary_signal_name = "corps"
        self.primary_reference_y = None
        self.best_signal_y = None
        self.best_signal_frame = None
        self.best_body_y_after_start = None
        self.body_downward_streak = 0
        self.last_signal_y = None
        self.last_signal_speed = None
        self.last_signal_amplitude = None

        self.image_debut_mouvement = None
        self.image_position_finale = None
        self.image_fin_essai = None
        self.frame_faute_redescente = None

        self.upward_streak = 0
        self.downward_streak = 0
        self.top_candidate_streak = 0
        self.top_hold_streak = 0

        self.faute_redescente = False
        self.faute_genoux_non_verrouilles = False
        self.final_knee_angle = None
        self.genoux_verrouilles = None
        self.lockout_evalue = False

        self.last_knee_left = None
        self.last_knee_right = None
        self.last_knee_min = None
        self.last_scale_px = 100.0
        self.last_body_center_y = None
        self.last_body_amplitude = None
        self.last_foot_fault = False
        self.dynamic_foot_threshold_set = False

        self.add_event(-1, "Analyse deadlift demarree")
        if self.face_bar_model is not None:
            self.add_event(
                -1,
                f"Suivi barre face actif via {os.path.basename(self.face_bar_model_path)}.",
            )
        if self.bar_tracking_enabled:
            self.add_event(
                -1,
                "Deadlift multi-vues : corps sur vue 1, trajectoire de barre via disque sur vue laterale 1.",
            )
        else:
            self.add_event(
                -1,
                "Deadlift 1 vue : analyse du corps seulement, trajectoire barre inactive.",
            )

    def process_frame(self, frames: List, source_frame_index: int):
        """Analyse un paquet de frames pour le deadlift.

        La vue face traite le signal corps, les pieds et les redescendes visibles
        de face. La vue latérale suit uniquement le disque pour la remontée et
        calcule sa vitesse avec lateral_fps, indépendamment du timing de la face.
        """

        self.indice_image = source_frame_index
        vues_annotees = [f.copy() for f in frames if f is not None]
        video_corps = vues_annotees[0] if vues_annotees else None
        video_barre = vues_annotees[1] if self.bar_tracking_enabled and len(vues_annotees) > 1 else None

        points = None
        if video_corps is not None:
            points = self.predict_pose_points(video_corps)
            if points is not None:
                dessiner_squelette(video_corps, points)
            self._update_face_bar(video_corps)

        # Prédire et dessiner le squelette sur la vidéo latérale avec le modèle latéral
        if video_barre is not None and self.lateral_athlete_model is not None:
            lateral_points = self._predict_lateral_athlete_points(video_barre)
            if lateral_points is not None:
                dessiner_squelette_lateral(video_barre, lateral_points)

        body_y_lisse, hip_width_lisse = self._update_body_metrics(points)
        bar_y_lisse = None
        if self.bar_tracking_enabled and video_barre is not None:
            bar_y_lisse = self._update_bar_metrics(video_barre)

        active_name, signal_y, signal_speed, signal_reference = self._select_signal(
            body_y_lisse,
            bar_y_lisse,
        )

        scale_px = self._compute_scale(hip_width_lisse)
        self.last_scale_px = scale_px

        start_disp_threshold = max(
            DEADLIFT_START_DISPLACEMENT_MIN_PX,
            DEADLIFT_START_DISPLACEMENT_RATIO_HIPWIDTH * scale_px,
        )
        start_speed_threshold = max(
            DEADLIFT_START_VELOCITY_MIN_PX,
            DEADLIFT_START_VELOCITY_RATIO_HIPWIDTH * scale_px,
        )
        top_min_ascent = max(
            DEADLIFT_TOP_MIN_ASCENT_MIN_PX,
            DEADLIFT_TOP_MIN_ASCENT_RATIO_HIPWIDTH * scale_px,
        )
        top_stability_threshold = max(
            DEADLIFT_TOP_STABILITY_MIN_PX,
            DEADLIFT_TOP_STABILITY_RATIO_HIPWIDTH * scale_px,
        )
        top_plateau_threshold = max(
            DEADLIFT_TOP_PLATEAU_MIN_PX,
            DEADLIFT_TOP_PLATEAU_RATIO_HIPWIDTH * scale_px,
        )
        redesc_threshold = max(
            DEADLIFT_REDESCENT_MIN_PX,
            DEADLIFT_REDESCENT_RATIO_HIPWIDTH * scale_px,
        )
        if self.bar_tracking_enabled:
            self._update_lateral_deadlift_finish(
                top_stability_threshold=top_stability_threshold,
                top_plateau_threshold=top_plateau_threshold,
            )
        self._update_face_redescent_fault(
            body_y_lisse=body_y_lisse,
            top_stability_threshold=top_stability_threshold,
            redesc_threshold=redesc_threshold,
        )

        if self.etat in {"attente", "setup"}:
            self._update_setup_state(
                active_name=active_name,
                signal_y=signal_y,
                signal_speed=signal_speed,
                signal_reference=signal_reference,
                hip_width_lisse=hip_width_lisse,
                start_disp_threshold=start_disp_threshold,
                start_speed_threshold=start_speed_threshold,
            )
        elif self.etat == "montee":
            self._update_montee_state(
                signal_y=signal_y,
                signal_speed=signal_speed,
                top_min_ascent=top_min_ascent,
                top_stability_threshold=top_stability_threshold,
                top_plateau_threshold=top_plateau_threshold,
                redesc_threshold=redesc_threshold,
            )
        elif self.etat == "position_finale_candidate":
            self._update_top_candidate_state(
                signal_y=signal_y,
                signal_speed=signal_speed,
                top_stability_threshold=top_stability_threshold,
                top_plateau_threshold=top_plateau_threshold,
                redesc_threshold=redesc_threshold,
            )

        calibration_states = {"attente", "setup"}
        active_states = {"montee", "position_finale_candidate"}
        if video_corps is not None:
            faute_pieds = self.detecteur_pieds.update(
                frame=video_corps,
                etat=self.etat,
                points=points,
                indice_image=self.indice_image,
                ajouter_evenement=self.add_event,
                calibration_states=calibration_states,
                active_states=active_states,
            )
        else:
            faute_pieds = False
        self.last_foot_fault = faute_pieds

        if DEBUG_PIEDS and self.indice_image % DEBUG_PIEDS_EVERY == 0:
            sy = f"{signal_y:.1f}" if signal_y is not None else "-"
            ss = f"{signal_speed:+.2f}" if signal_speed is not None else "-"
            print(
                f"[DEADLIFT] frame={self.indice_image} etat={self.etat} "
                f"source={active_name} signal_y={sy} signal_speed={ss} "
                f"knee_min={self.last_knee_min if self.last_knee_min is not None else '-'} "
                f"faute_pieds={faute_pieds}"
            )

        verdict_global, sections = self._build_dashboard_sections()
        if video_corps is not None:
            self._draw_overlay_header(
                video_corps,
                "Deadlift",
                f"Etat: {self.etat} | Verdict: {verdict_global}",
                (0, 220, 140),
            )

        return self.compose_output(
            vues_annotees,
            ips_txt=self.format_fps_text(),
            indice_image=self.indice_image,
            t_sec=self.indice_image / self.fps,
            etat=self.etat,
            verdict_global=verdict_global,
            sections=sections,
        )

    def _update_body_metrics(self, points) -> Tuple[Optional[float], Optional[float]]:
        """Met à jour les mesures corps utilisées par la machine d'état.

        La méthode calcule un Y corps moyen depuis épaules/bassin, lisse ce Y,
        met à jour la vitesse verticale du corps, la largeur de bassin et les
        angles de genoux visibles.
        """

        if points is None:
            self.last_knee_left = None
            self.last_knee_right = None
            self.last_knee_min = None
            self.last_body_speed = None
            return None, float(median(self.hip_width_buffer)) if self.hip_width_buffer else None

        y_epaules, _ = y_epaules_et_largeur(points)
        y_bassin, largeur_bassin = y_bassin_et_largeur(points)
        if largeur_bassin is not None:
            self.hip_width_buffer.append(largeur_bassin)

        valeurs_y = [value for value in (y_epaules, y_bassin) if value is not None]
        body_y = float(sum(valeurs_y) / len(valeurs_y)) if valeurs_y else None
        if body_y is not None:
            self.body_y_buffer.append(body_y)
            if self.etat in {"attente", "setup"}:
                self.body_setup_buffer.append(body_y)
                if len(self.body_setup_buffer) >= max(6, DEADLIFT_SETUP_FRAMES // 2):
                    self.body_reference_y = float(median(self.body_setup_buffer))
                    if self.etat == "attente":
                        self.etat = "setup"
                        self.add_event(
                            self.indice_image,
                            "Setup detecte : reference corps stabilisee",
                        )

        body_y_lisse = (
            float(median(self.body_y_buffer))
            if len(self.body_y_buffer) >= max(3, FENETRE_LISSAGE // 2)
            else None
        )
        self.last_body_speed = (
            body_y_lisse - self.last_body_y
            if body_y_lisse is not None and self.last_body_y is not None
            else None
        )
        if body_y_lisse is not None:
            self.last_body_y = body_y_lisse
            if self.body_reference_y is not None:
                self.last_body_amplitude = self.body_reference_y - body_y_lisse

        angle_g, angle_d = angles_genoux(points)
        if angle_g is not None:
            self.knee_left_buffer.append(angle_g)
        if angle_d is not None:
            self.knee_right_buffer.append(angle_d)
        self.last_knee_left = float(median(self.knee_left_buffer)) if self.knee_left_buffer else None
        self.last_knee_right = float(median(self.knee_right_buffer)) if self.knee_right_buffer else None
        self.last_knee_min = angle_genou_min_visible(points)

        return body_y_lisse, float(median(self.hip_width_buffer)) if self.hip_width_buffer else None

    def _update_face_bar(self, video_face) -> None:
        """Détecte et dessine la barre sur la vue face si barre_face.pt existe.

        Cette détection sert surtout à l'affichage et au diagnostic. Elle ne
        remplace pas le suivi du disque latéral pour les vitesses.
        """

        self.last_face_bar_points = None
        self.last_face_bar_center = None
        self.last_face_bar_detected = False
        if self.face_bar_model is None:
            return

        results = self.face_bar_model.predict(
            source=video_face,
            conf=self.conf,
            verbose=False,
        )
        result = results[0] if results else None
        points = choisir_personne_principale(result)
        self.last_face_bar_points = points
        if not points or len(points) < 2:
            return

        (x1, y1), (x2, y2) = points[0], points[1]
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            return

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.line(video_face, p1, p2, (255, 0, 255), 3)
        cv2.circle(video_face, p1, 6, (0, 0, 255), -1)
        cv2.circle(video_face, p2, 6, (0, 0, 255), -1)
        cv2.putText(
            video_face,
            "L",
            (p1[0] + 5, p1[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            video_face,
            "R",
            (p2[0] + 5, p2[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        self.last_face_bar_center = (
            int(round((x1 + x2) / 2.0)),
            int(round((y1 + y2) / 2.0)),
        )
        self.last_face_bar_detected = True

    def _predict_lateral_athlete_points(self, frame):
        """Prédit les points de pose de l'athlète sur la vue latérale.

        Le résultat sert uniquement au dessin du squelette latéral quand un
        modèle lateral_athlete est disponible.
        """
        if self.lateral_athlete_model is None:
            return None
        resultats = self.lateral_athlete_model.predict(
            source=frame,
            conf=self.conf,
            verbose=False,
        )
        result = resultats[0] if resultats else None
        return choisir_personne_principale(result)

    def _update_bar_metrics(self, video_barre) -> Optional[float]:
        """Suit le disque latéral et met à jour sa trajectoire.

        Cette méthode détecte le disque, lisse son Y, met à jour la phase
        latérale remontée/descente, enregistre les points de remontée et dessine
        la trajectoire verte. Elle est indépendante de la vue face.
        """

        disk_signal = self._predict_signal(
            self.disk_model,
            self.disk_target_class,
            self.disk_history,
            "disque",
            video_barre,
        )
        self.last_disk_signal = disk_signal

        if disk_signal.center is not None:
            draw_detection_signal(
                video_barre,
                disk_signal,
                (255, 180, 0),
                (255, 180, 0),
                "disque",
            )

        self.last_fused_signal = disk_signal
        if disk_signal.center is not None:
            self._draw_fused_signal(video_barre, disk_signal)
            self.bar_x_buffer.append(disk_signal.center[0])
            self.bar_y_buffer.append(disk_signal.center[1])
            if not self.lateral_disk_frozen:
                self.bar_path_points.append(disk_signal.center)
            self.disk_positions.append(disk_signal.center)
            self.last_bar_source = "disque"

            if disk_signal.box is not None:
                x1, y1, x2, y2 = disk_signal.box
                diameter_px = x2 - x1
                if self.cm_per_pixel is None and diameter_px > 0:
                    self.cm_per_pixel = 45.0 / diameter_px

            if self.last_bar_y is not None:
                (
                    self.phase_disk_laterale,
                    self.compteur_descente_disk,
                    self.compteur_remontee_disk,
                ) = phase_barre_depuis_vue_laterale(
                    disk_signal.center[1],
                    self.last_bar_y,
                    self.phase_disk_laterale,
                    self.compteur_descente_disk,
                    self.compteur_remontee_disk,
                )
                if (
                    self.phase_disk_laterale == "remontee"
                    and self.phase_disk_laterale_precedente != "remontee"
                    and not self.lateral_disk_frozen
                ):
                    self._reset_lateral_deadlift_phase()
                    self.image_debut_remontee_disk = max(
                        0,
                        self.indice_image - NB_IMAGES_CONSEC_BARRE_PHASE + 1,
                    )
                elif (
                    self.phase_disk_laterale == "descente"
                    and self.phase_disk_laterale_precedente == "remontee"
                    and not self.lateral_disk_frozen
                    and self.disk_remontee_frames
                ):
                    self.image_fin_remontee_disk = self.disk_remontee_frames[-1][0]
                    self.lateral_disk_frozen = True
                    self._update_lateral_average_speed()

                self.phase_disk_laterale_precedente = self.phase_disk_laterale

        bar_y_lisse = (
            float(median(self.bar_y_buffer))
            if len(self.bar_y_buffer) >= max(3, FENETRE_LISSAGE // 2)
            else None
        )
        self.last_bar_speed = (
            bar_y_lisse - self.last_bar_y
            if bar_y_lisse is not None and self.last_bar_y is not None
            else None
        )
        if bar_y_lisse is not None:
            self.last_bar_y = bar_y_lisse
            if self.etat in {"attente", "setup"}:
                self.bar_setup_buffer.append(bar_y_lisse)
                if len(self.bar_setup_buffer) >= max(6, DEADLIFT_SETUP_FRAMES // 2):
                    self.bar_reference_y = float(median(self.bar_setup_buffer))

        if (
            disk_signal.center is not None
            and self.image_debut_remontee_disk is not None
            and not self.lateral_disk_frozen
            and self.phase_disk_laterale == "remontee"
        ):
            self.disk_remontee.append(disk_signal.center)
            self.disk_remontee_frames.append((self.indice_image, disk_signal.center))

        tracer_trajectoire(video_barre, self.disk_remontee, (0, 255, 0))

        return bar_y_lisse

    def _predict_signal(
        self,
        model,
        target_class: Optional[str],
        history,
        source: str,
        frame,
    ) -> DetectionSignal:
        """Lance un modèle YOLO de suivi et retourne un DetectionSignal lissé."""

        if model is None:
            return DetectionSignal(source=source)
        results = model.predict(source=frame, conf=BARBELL_CONF_THRES, verbose=False)
        result = results[0] if results else None
        return tracked_detection_signal(
            result,
            model.names,
            target_class,
            history,
            source,
        )

    def _draw_fused_signal(self, image, signal: DetectionSignal) -> None:
        """Dessine le signal disque principal utilisé par le deadlift."""

        if signal.box is not None:
            x1, y1, x2, y2 = signal.box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        if signal.center is not None:
            cv2.circle(image, signal.center, 6, (255, 0, 255), -1)

    def _select_signal(
        self,
        body_y_lisse: Optional[float],
        bar_y_lisse: Optional[float],
    ) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
        """Choisit si la machine d'état suit le corps ou le disque.

        Au setup, le disque est préféré s'il est disponible et calibré; sinon le
        corps sert de repli. Une fois le mouvement commencé, la source principale
        est conservée pour éviter les bascules instables.
        """

        if self.etat in {"attente", "setup"}:
            if self.bar_tracking_enabled and bar_y_lisse is not None and self.bar_reference_y is not None:
                return "disque", bar_y_lisse, self.last_bar_speed, self.bar_reference_y
            return "corps", body_y_lisse, self.last_body_speed, self.body_reference_y

        if self.primary_signal_name == "disque":
            return "disque", bar_y_lisse, self.last_bar_speed, self.primary_reference_y
        return "corps", body_y_lisse, self.last_body_speed, self.primary_reference_y

    def _compute_scale(self, hip_width_lisse: Optional[float]) -> float:
        """Calcule une échelle en pixels pour les seuils dynamiques."""

        if hip_width_lisse is not None:
            return float(max(hip_width_lisse, 40.0))
        if self.last_fused_signal.box is not None:
            x1, y1, x2, y2 = self.last_fused_signal.box
            return float(max(40, x2 - x1, y2 - y1))
        return 100.0

    def _reset_lateral_deadlift_phase(self) -> None:
        """Réinitialise les données latérales avant une nouvelle remontée disque."""

        self.disk_remontee.clear()
        self.disk_remontee_frames.clear()
        self.speed_remontee = None
        self.speed_remontee_details = None
        self.image_debut_remontee_disk = None
        self.image_fin_remontee_disk = None
        self.lateral_disk_frozen = False
        self.lateral_top_streak = 0

    def _calculate_lateral_ascent_speed(self):
        """Calcule la vitesse moyenne de remontée du deadlift.

        La méthode prend le point disque le plus bas et le point le plus haut de
        la remontée latérale, convertit la distance avec le disque de 45 cm, puis
        divise par le temps basé sur le FPS de la vidéo latérale.
        """

        if self.cm_per_pixel is None or len(self.disk_remontee_frames) < 2:
            return None

        bottom_frame, bottom_point = max(
            self.disk_remontee_frames,
            key=lambda item: item[1][1],
        )
        top_frame, top_point = min(
            self.disk_remontee_frames,
            key=lambda item: item[1][1],
        )
        delta_px = abs(bottom_point[1] - top_point[1])
        duration_s = abs(top_frame - bottom_frame) / self.lateral_fps
        if duration_s <= 0:
            return None

        distance_cm = delta_px * self.cm_per_pixel
        return {
            "speed_cm_s": distance_cm / duration_s,
            "distance_cm": distance_cm,
            "delta_px": delta_px,
            "duration_s": duration_s,
            "bottom_frame": bottom_frame,
            "top_frame": top_frame,
            "bottom_y": bottom_point[1],
            "top_y": top_point[1],
        }

    def _update_lateral_average_speed(self) -> None:
        """Stocke la vitesse latérale une fois la remontée terminée."""

        if self.speed_remontee is not None:
            return

        details = self._calculate_lateral_ascent_speed()
        if details is None:
            return

        self.speed_remontee_details = details
        self.speed_remontee = details["speed_cm_s"]
        self.add_event(
            max(details["bottom_frame"], details["top_frame"]),
            f"Vitesse moyenne remontee deadlift: {self.speed_remontee:.2f} cm/s",
        )

    def _update_lateral_deadlift_finish(
        self,
        *,
        top_stability_threshold: float,
        top_plateau_threshold: float,
    ) -> None:
        """Détecte une position haute stable à partir du disque latéral."""

        if (
            not self.bar_tracking_enabled
            or self.lateral_disk_frozen
            or self.image_debut_remontee_disk is None
            or not self.disk_remontee_frames
            or self.last_bar_y is None
            or self.last_bar_speed is None
        ):
            return

        best_y = min(point[1] for _, point in self.disk_remontee_frames)
        stable_at_top = (
            abs(self.last_bar_speed) <= top_stability_threshold
            and abs(self.last_bar_y - best_y) <= top_plateau_threshold
        )
        if stable_at_top:
            self.lateral_top_streak += 1
        else:
            self.lateral_top_streak = 0

        if self.lateral_top_streak >= DEADLIFT_TOP_HOLD_FRAMES:
            self.image_fin_remontee_disk = self.disk_remontee_frames[-1][0]
            self.lateral_disk_frozen = True
            self._update_lateral_average_speed()

    def _update_face_redescent_fault(
        self,
        *,
        body_y_lisse: Optional[float],
        top_stability_threshold: float,
        redesc_threshold: float,
    ) -> None:
        """Détecte une faute de redescente depuis le signal corps de face."""

        if self.etat not in {"montee", "position_finale_candidate"}:
            return
        if body_y_lisse is None or self.last_body_speed is None:
            return

        if self.best_body_y_after_start is None or body_y_lisse < self.best_body_y_after_start:
            self.best_body_y_after_start = body_y_lisse
            self.body_downward_streak = 0
            return

        redescente = (
            self.last_body_speed >= top_stability_threshold
            and (body_y_lisse - self.best_body_y_after_start) >= redesc_threshold
        )
        if redescente:
            self.body_downward_streak += 1
        else:
            self.body_downward_streak = 0

        if self.body_downward_streak >= DEADLIFT_REDESCENT_CONSEC_FRAMES:
            self.faute_redescente = True
            self.frame_faute_redescente = (
                self.indice_image - DEADLIFT_REDESCENT_CONSEC_FRAMES + 1
            )
            self.image_fin_essai = self.frame_faute_redescente
            self.etat = "termine"
            self.add_event(
                self.indice_image,
                "Essai refuse : redescente detectee sur la vue face.",
            )

    def _update_setup_state(
        self,
        *,
        active_name: str,
        signal_y: Optional[float],
        signal_speed: Optional[float],
        signal_reference: Optional[float],
        hip_width_lisse: Optional[float],
        start_disp_threshold: float,
        start_speed_threshold: float,
    ) -> None:
        """Passe du setup à la montée quand le signal choisi monte clairement."""

        if signal_y is None or signal_speed is None or signal_reference is None:
            return

        displacement = signal_reference - signal_y
        if displacement >= start_disp_threshold and signal_speed <= -start_speed_threshold:
            self.upward_streak += 1
        else:
            self.upward_streak = 0

        if self.upward_streak >= DEADLIFT_START_CONSEC_FRAMES:
            self.etat = "montee"
            self.primary_signal_name = active_name
            self.primary_reference_y = signal_reference
            self.image_debut_mouvement = (
                self.indice_image - DEADLIFT_START_CONSEC_FRAMES + 1
            )
            self.best_signal_y = signal_y
            self.best_signal_frame = self.indice_image
            self.best_body_y_after_start = self.last_body_y
            self.body_downward_streak = 0
            self.last_signal_y = signal_y
            self.last_signal_speed = signal_speed
            self.downward_streak = 0
            self.top_candidate_streak = 0
            self.top_hold_streak = 0
            self.add_event(
                self.indice_image,
                f"Debut du mouvement detecte depuis le {active_name} (frame {self.image_debut_mouvement})",
            )
            if hip_width_lisse is not None and not self.dynamic_foot_threshold_set:
                self.detecteur_pieds.deplacement_threshold_px = max(
                    FOOT_DISPLACEMENT_THRESHOLD,
                    DEADLIFT_FOOT_DISPLACEMENT_RATIO_HIPWIDTH * hip_width_lisse,
                )
                self.dynamic_foot_threshold_set = True

    def _update_montee_state(
        self,
        *,
        signal_y: Optional[float],
        signal_speed: Optional[float],
        top_min_ascent: float,
        top_stability_threshold: float,
        top_plateau_threshold: float,
        redesc_threshold: float,
    ) -> None:
        """Met à jour l'état de montée.

        La méthode garde le meilleur point atteint, détecte une redescente
        prématurée et cherche une position finale candidate lorsque le signal se
        stabilise au sommet.
        """

        if signal_y is None:
            return

        if self.best_signal_y is None or signal_y < self.best_signal_y:
            self.best_signal_y = signal_y
            self.best_signal_frame = self.indice_image

        if self.primary_reference_y is not None and self.best_signal_y is not None:
            self.last_signal_amplitude = self.primary_reference_y - self.best_signal_y

        if (
            signal_speed is not None
            and self.best_signal_y is not None
            and signal_speed >= top_stability_threshold
            and (signal_y - self.best_signal_y) >= redesc_threshold
        ):
            self.downward_streak += 1
        else:
            self.downward_streak = 0

        if self.downward_streak >= DEADLIFT_REDESCENT_CONSEC_FRAMES:
            self.faute_redescente = True
            self.frame_faute_redescente = (
                self.indice_image - DEADLIFT_REDESCENT_CONSEC_FRAMES + 1
            )
            self.image_fin_essai = self.frame_faute_redescente
            self.etat = "termine"
            self.add_event(
                self.indice_image,
                "Essai refuse : la barre redescend avant la position finale.",
            )
            return

        near_top = (
            self.last_signal_amplitude is not None
            and self.last_signal_amplitude >= top_min_ascent
            and signal_speed is not None
            and abs(signal_speed) <= top_stability_threshold
            and self.best_signal_y is not None
            and abs(signal_y - self.best_signal_y) <= top_plateau_threshold
        )
        if near_top:
            self.top_candidate_streak += 1
        else:
            self.top_candidate_streak = 0

        if self.top_candidate_streak >= DEADLIFT_TOP_CANDIDATE_FRAMES:
            self.etat = "position_finale_candidate"
            self.image_position_finale = (
                self.indice_image - DEADLIFT_TOP_CANDIDATE_FRAMES + 1
            )
            self.top_hold_streak = self.top_candidate_streak
            self.final_lockout_window.clear()
            self._append_lockout_sample()
            self.add_event(
                self.indice_image,
                f"Position finale candidate detectee (frame {self.image_position_finale})",
            )

    def _update_top_candidate_state(
        self,
        *,
        signal_y: Optional[float],
        signal_speed: Optional[float],
        top_stability_threshold: float,
        top_plateau_threshold: float,
        redesc_threshold: float,
    ) -> None:
        """Confirme ou rejette la position finale candidate."""

        self._append_lockout_sample()
        if signal_y is None:
            return

        if (
            signal_speed is not None
            and self.best_signal_y is not None
            and signal_speed >= top_stability_threshold
            and (signal_y - self.best_signal_y) >= redesc_threshold
            and self.top_hold_streak < DEADLIFT_TOP_HOLD_FRAMES
        ):
            self.faute_redescente = True
            self.frame_faute_redescente = self.indice_image
            self.image_fin_essai = self.indice_image
            self.etat = "termine"
            self.add_event(
                self.indice_image,
                "Essai refuse : la barre redescend avant la position finale.",
            )
            return

        if (
            signal_speed is not None
            and self.best_signal_y is not None
            and abs(signal_speed) <= top_stability_threshold
            and abs(signal_y - self.best_signal_y) <= top_plateau_threshold
        ):
            self.top_hold_streak += 1

        if self.top_hold_streak >= DEADLIFT_TOP_HOLD_FRAMES:
            self.image_fin_essai = self.indice_image
            self.etat = "termine"
            self.add_event(
                self.indice_image,
                f"Fin d'essai detectee (frame {self.image_fin_essai})",
            )
            self._evaluate_lockout()

    def _append_lockout_sample(self) -> None:
        """Ajoute un échantillon d'angle de genou pour le lockout."""

        visibles = [
            angle
            for angle in (self.last_knee_left, self.last_knee_right)
            if angle is not None
        ]
        if visibles:
            self.final_lockout_window.append(float(min(visibles)))

    def _evaluate_lockout(self) -> None:
        """Évalue le verrouillage des genoux.

        Le résultat est actuellement affiché dans le tableau de bord, mais il ne
        participe pas encore au verdict final.
        """

        if self.lockout_evalue:
            return
        self.lockout_evalue = True
        valeurs = list(self.final_lockout_window)
        if valeurs:
            self.final_knee_angle = float(median(valeurs))
            self.genoux_verrouilles = (
                self.final_knee_angle >= DEADLIFT_LOCKOUT_KNEE_ANGLE
            )
            if self.genoux_verrouilles:
                self.add_event(
                    self.indice_image,
                    "Verrouillage genoux OK en fin de mouvement.",
                )
            else:
                self.faute_genoux_non_verrouilles = True
                self.add_event(
                    self.indice_image,
                    "Essai refuse : genoux non verrouilles a la fin du mouvement.",
                )
        else:
            self.genoux_verrouilles = None
            self.add_event(
                self.indice_image,
                "Verrouillage genoux : donnees insuffisantes pour conclure.",
            )

    def _build_dashboard_sections(self):
        """Construit les sections du tableau de bord deadlift et le verdict."""

        raisons = []
        if self.faute_redescente:
            raisons.append("la barre redescend avant la position finale")
        if self.last_foot_fault:
            raisons.append("deplacement du pied detecte pendant l'essai")

        if self.etat == "termine" and not raisons:
            verdict_global = "Essai valide."
        elif raisons:
            verdict_global = "Essai refuse : " + "; ".join(raisons) + "."
        else:
            verdict_global = "Analyse en cours."

        sections = [
            (
                "REPÈRES TEMPORELS (VUE CORPS)",
                [
                    f"Setup détecté        : {'Oui' if self.body_reference_y is not None else 'Non'}",
                    f"Début mouvement      : {self.image_debut_mouvement if self.image_debut_mouvement is not None else '—'}",
                    f"Position finale cand.: {self.image_position_finale if self.image_position_finale is not None else '—'}",
                    f"Fin essai            : {self.image_fin_essai if self.image_fin_essai is not None else '—'}",
                ],
            ),
            (
                "DÉTECTION BARRE VUE FACE",
                [
                    f"Modèle actif         : {'Oui' if self.face_bar_model is not None else 'Non'}",
                    f"Fichier              : {os.path.basename(self.face_bar_model_path) if self.face_bar_model_path else '—'}",
                    f"Barre détectée       : {'Oui' if self.last_face_bar_detected else 'Non'}",
                    f"Nb keypoints barre   : {len(self.last_face_bar_points) if self.last_face_bar_points is not None else 0}",
                    f"Centre barre face    : {self.last_face_bar_center if self.last_face_bar_center is not None else '—'}",
                ],
            ),
            (
                "REPÈRES TEMPORELS (VUE LATÉRALE)",
                [
                    f"Suivi disque actif   : {'Oui' if self.bar_tracking_enabled else 'Non'}",
                    f"Phase disque latérale: {self.phase_disk_laterale if self.phase_disk_laterale is not None else '—'}",
                    f"Début remontée disque: {self.image_debut_remontee_disk if self.image_debut_remontee_disk is not None else '—'}",
                    f"Fin remontée disque  : {self.image_fin_remontee_disk if self.image_fin_remontee_disk is not None else '—'}",
                    f"Source principale    : {self.primary_signal_name}",
                    f"Référence disque (px): {f'{self.bar_reference_y:.1f}' if self.bar_reference_y is not None else '—'}",
                    f"Frame faute redesc.  : {self.frame_faute_redescente if self.frame_faute_redescente is not None else '—'}",
                ],
            ),
            (
                "DONNÉES LISSÉES",
                [
                    f"Y corps (px)         : {f'{self.last_body_y:.1f}' if self.last_body_y is not None else '—'}",
                    f"Vitesse corps (px/f) : {f'{self.last_body_speed:+.2f}' if self.last_body_speed is not None else '—'}",
                    f"Amplitude corps (px) : {f'{self.last_body_amplitude:.1f}' if self.last_body_amplitude is not None else '—'}",
                    f"Largeur bassin (px)  : {f'{median(self.hip_width_buffer):.1f}' if self.hip_width_buffer else '—'}",
                ],
            ),
        ]

        if self.bar_tracking_enabled:
            sections.append(
                (
                    "ANALYSE TRAJECTOIRE BARRE",
                    [
                        "Source               : Vue latérale disque",
                        f"Points trajectoire   : {len(self.bar_path_points)}",
                        f"Y disque lissé (px)  : {f'{self.last_bar_y:.1f}' if self.last_bar_y is not None else '—'}",
                        f"Vitesse disque (px/f): {f'{self.last_bar_speed:+.2f}' if self.last_bar_speed is not None else '—'}",
                        f"Redescente détectée  : {'Oui' if self.faute_redescente else 'Non'}",
                        f"Verdict trajectoire  : {'FAUTE' if self.faute_redescente else ('OK' if self.etat == 'termine' else 'EN COURS')}",
                    ],
                )
            )
        else:
            sections.append(
                (
                    "ANALYSE TRAJECTOIRE BARRE",
                    [
                        "Source               : Vue corps seulement",
                        "Suivi latéral disque : Inactif en 1 vue",
                    ],
                )
            )

        sections.extend(
            [
                (
                    "ANALYSE GENOUX",
                    [
                        f"Genou gauche lissé   : {f'{self.last_knee_left:.1f}°' if self.last_knee_left is not None else '—'}",
                        f"Genou droit lissé    : {f'{self.last_knee_right:.1f}°' if self.last_knee_right is not None else '—'}",
                        f"Min genou visible    : {f'{self.last_knee_min:.1f}°' if self.last_knee_min is not None else '—'}",
                        f"Seuil lockout        : {DEADLIFT_LOCKOUT_KNEE_ANGLE:.1f}°",
                        f"Verdict verrouillage : {'FAUTE' if self.faute_genoux_non_verrouilles else ('OK' if self.genoux_verrouilles else '—')}",
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
                (
                    "SEUILS UTILISÉS",
                    [
                        f"Échelle courante (px): {self.last_scale_px:.1f}",
                        f"Start / lift-off     : max({DEADLIFT_START_DISPLACEMENT_MIN_PX:.1f}, {DEADLIFT_START_DISPLACEMENT_RATIO_HIPWIDTH:.3f}*échelle)",
                        f"Top stable           : max({DEADLIFT_TOP_STABILITY_MIN_PX:.1f}, {DEADLIFT_TOP_STABILITY_RATIO_HIPWIDTH:.3f}*échelle)",
                        f"Redescente disque    : max({DEADLIFT_REDESCENT_MIN_PX:.1f}, {DEADLIFT_REDESCENT_RATIO_HIPWIDTH:.3f}*échelle)",
                    ],
                ),
            ]
        )

        if self.bar_tracking_enabled:
            vitesse_lignes = [
                f"FPS latéral      : {self.lateral_fps:.2f}",
                f"Échelle          : {self.cm_per_pixel:.4f} cm/px" if self.cm_per_pixel is not None else "Échelle          : —",
                f"Vitesse remontée moyenne : {self.speed_remontee:.2f} cm/s" if self.speed_remontee is not None else "Vitesse remontée moyenne : —",
            ]
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
            sections.append(
                (
                    "VITESSE DISQUE",
                    vitesse_lignes,
                )
            )

        return verdict_global, sections
