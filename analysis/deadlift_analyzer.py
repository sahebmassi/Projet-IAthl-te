import os
from collections import deque
from statistics import median
from typing import List, Optional, Tuple

import cv2

from .barbell_tracking import (
    DetectionSignal,
    draw_detection_signal,
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
    FENETRE_LISSAGE,
    FOOT_CALIBRATION_FRAMES,
    FOOT_DISPLACEMENT_THRESHOLD,
    FOOT_FLOW_ACCUMULATION_WINDOW,
    FOOT_GRACE_PERIOD_FRAMES,
    FOOT_PERSISTANCE_FRAMES,
)
from .foot_detector import DetecteurPiedsRobuste
from .geometry import angles_genoux, angle_genou_min_visible, y_bassin_et_largeur, y_epaules_et_largeur
from .pose import choisir_personne_principale, dessiner_squelette, dessiner_squelette_lateral


class DeadliftAnalyzer(BaseMovementAnalyzer):
    dashboard_title = "TABLEAU DE BORD - DEADLIFT"

    def __init__(self, **kwargs):
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
        self.last_face_bar_points = None
        self.last_face_bar_center = None
        self.last_face_bar_detected = False

        self.hip_width_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.knee_left_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.knee_right_buffer = deque(maxlen=FENETRE_LISSAGE)
        self.final_lockout_window = deque(maxlen=DEADLIFT_TOP_HOLD_FRAMES)
        self.bar_path_points = []

        self.detecteur_pieds = DetecteurPiedsRobuste(
            nb_images_calibration=FOOT_CALIBRATION_FRAMES,
            flow_accumulation_window=FOOT_FLOW_ACCUMULATION_WINDOW,
            deplacement_threshold_px=FOOT_DISPLACEMENT_THRESHOLD,
            nb_images_persistance=FOOT_PERSISTANCE_FRAMES,
            grace_period_frames=FOOT_GRACE_PERIOD_FRAMES,
            ignorer_apres_fin=True,
            debug=DEBUG_PIEDS,
        )

        self.indice_image = -1
        self.etat = "attente"

        self.primary_signal_name = "corps"
        self.primary_reference_y = None
        self.best_signal_y = None
        self.best_signal_frame = None
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
        self.indice_image = source_frame_index
        vues_annotees = [frame.copy() for frame in frames]
        video_corps = vues_annotees[0]
        video_barre = vues_annotees[1] if self.bar_tracking_enabled and len(vues_annotees) > 1 else None

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
        active_states = {"montee", "position_finale_candidate", "termine"}
        faute_pieds = self.detecteur_pieds.update(
            frame=video_corps,
            etat=self.etat,
            points=points,
            indice_image=self.indice_image,
            ajouter_evenement=self.add_event,
            calibration_states=calibration_states,
            active_states=active_states,
        )
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
        """Prédire les points de pose sur la vidéo latérale avec le modèle latéral."""
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
            self.bar_path_points.append(disk_signal.center)
            self.last_bar_source = "disque"

        tracer_trajectoire(video_barre, self.bar_path_points, (0, 255, 0))

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

        return bar_y_lisse

    def _predict_signal(
        self,
        model,
        target_class: Optional[str],
        history,
        source: str,
        frame,
    ) -> DetectionSignal:
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
        if self.etat in {"attente", "setup"}:
            if self.bar_tracking_enabled and bar_y_lisse is not None and self.bar_reference_y is not None:
                return "disque", bar_y_lisse, self.last_bar_speed, self.bar_reference_y
            return "corps", body_y_lisse, self.last_body_speed, self.body_reference_y

        if self.primary_signal_name == "disque":
            return "disque", bar_y_lisse, self.last_bar_speed, self.primary_reference_y
        return "corps", body_y_lisse, self.last_body_speed, self.primary_reference_y

    def _compute_scale(self, hip_width_lisse: Optional[float]) -> float:
        if hip_width_lisse is not None:
            return float(max(hip_width_lisse, 40.0))
        if self.last_fused_signal.box is not None:
            x1, y1, x2, y2 = self.last_fused_signal.box
            return float(max(40, x2 - x1, y2 - y1))
        return 100.0

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
        if signal_y is None:
            return

        if self.best_signal_y is None or signal_y < self.best_signal_y:
            self.best_signal_y = signal_y
            self.best_signal_frame = self.indice_image

        if self.primary_reference_y is not None and self.best_signal_y is not None:
            self.last_signal_amplitude = self.primary_reference_y - self.best_signal_y

        if (
            self.primary_signal_name == "disque"
            and signal_speed is not None
            and self.best_signal_y is not None
            and signal_speed >= top_stability_threshold
            and (signal_y - self.best_signal_y) >= redesc_threshold
        ):
            self.downward_streak += 1
        else:
            self.downward_streak = 0

        if (
            self.primary_signal_name == "disque"
            and self.downward_streak >= DEADLIFT_REDESCENT_CONSEC_FRAMES
        ):
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
        self._append_lockout_sample()
        if signal_y is None:
            return

        if (
            self.primary_signal_name == "disque"
            and signal_speed is not None
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
        visibles = [
            angle
            for angle in (self.last_knee_left, self.last_knee_right)
            if angle is not None
        ]
        if visibles:
            self.final_lockout_window.append(float(min(visibles)))

    def _evaluate_lockout(self) -> None:
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
        raisons = []
        if self.faute_redescente:
            raisons.append("la barre redescend avant la position finale")
        if self.faute_genoux_non_verrouilles:
            raisons.append("genoux non verrouilles a la fin du mouvement")
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
        return verdict_global, sections
