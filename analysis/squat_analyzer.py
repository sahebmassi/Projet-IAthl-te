from collections import deque
from statistics import median
from typing import List

from .barbell_tracking import (
    draw_detection_signal,
    phase_barre_depuis_vue_face,
    phase_barre_depuis_vue_laterale,
    resolve_barbell_target_class,
    tracked_detection_signal,
    tracer_trajectoire,
    comparer_trajectoires,
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
    DUREE_HAUT_SEC,
    DUREE_MIN_DESCENTE_A_60_IPS,
    DUREE_VERROUILLAGE_SEC,
    FENETRE_LISSAGE,
    FOOT_CALIBRATION_FRAMES,
    FOOT_DISPLACEMENT_THRESHOLD,
    FOOT_FLOW_ACCUMULATION_WINDOW,
    FOOT_GRACE_PERIOD_FRAMES,
    FOOT_PERSISTANCE_FRAMES,
    K_MIN_DIP_SEC,
    NB_IMAGES_CONSEC_DESCENTE,
    NB_IMAGES_CONSEC_REMONTEE,
    NB_IMAGES_CONSEC_BARRE_PHASE,
    SEUIL_ANGLE_GENOU_PROFONDEUR,
    SEUIL_STABILITE_RATIO_LARGEUR_BASSIN,
    SEUIL_VERROUILLAGE_GENOU,
    SEUIL_VITESSE_RATIO_LARGEUR_BASSIN,
    X_IGNORE_AFTER_ASCENT_SEC,
)
from .foot_detector import DetecteurPiedsRobuste
from .geometry import angle_moyen_genoux, y_bassin_et_largeur, y_moyenne_genoux
from .pose import dessiner_squelette


class SquatAnalyzer(BaseMovementAnalyzer):
    dashboard_title = "TABLEAU DE BORD - SQUAT"

    def __init__(self, **kwargs):
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
        self.trajectoire_ok = None
        self.trajectoire_stats = None
        self.barbell_target_class = resolve_barbell_target_class(
            self.barbell_model_path,
            self.barbell_model.names,
            preferred_kind="auto",
        )

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
        self.y_bassin_max = None
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
            ignorer_apres_fin=True,
            debug=DEBUG_PIEDS,
        )

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
        self.indice_image = source_frame_index
        vues_annotees = [frame.copy() for frame in frames]
        video_face = vues_annotees[0]
        video_barre = vues_annotees[1] if len(vues_annotees) > 1 else video_face

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
                        self.traj_descente.clear()
                        self.traj_remontee.clear()
                        self.hist_barbell.clear()
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
                            if (
                                len(vues_annotees) > 1
                                and self.image_fin_remontee_barre is None
                            ):
                                self.image_fin_remontee_barre = self.indice_image
                            raison = (
                                "genoux verrouilles"
                                if self.suite_verrouillage >= self.verrouillage_images
                                else "retour en haut"
                            )
                            self.add_event(
                                self.indice_image,
                                f"Fin de remontee ({raison}, frame {self.image_fin_remontee})",
                            )
                            self.trajectoire_ok, self.trajectoire_stats = comparer_trajectoires(
                                self.traj_descente,
                                self.traj_remontee,
                                largeur_bassin_lisse,
                            )
                            if self.trajectoire_ok is None:
                                self.add_event(
                                    self.indice_image,
                                    "Trajectoire : donnees insuffisantes",
                                )
                            else:
                                self.add_event(
                                    self.indice_image,
                                    "Trajectoire : "
                                    f"{'OK' if self.trajectoire_ok else 'FAUTE'} "
                                    f"(ecart moyen {self.trajectoire_stats['ecart_moyen']:.1f}px / "
                                    f"seuil {self.trajectoire_stats['seuil']:.1f}px)",
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

            phase_barre_face = phase_barre_depuis_vue_face(
                self.indice_image,
                self.image_debut_descente,
                self.image_debut_remontee,
                self.image_fin_remontee,
            )

            if len(vues_annotees) > 1:
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
                phase_barre = (
                    self.phase_barre_laterale
                    if self.phase_barre_laterale is not None
                    else phase_barre_face
                )
            else:
                phase_barre = phase_barre_face

            if phase_barre == "descente":
                self.traj_descente.append(barbell_signal.center)
            elif phase_barre == "remontee":
                self.traj_remontee.append(barbell_signal.center)

        tracer_trajectoire(video_barre, self.traj_descente, (0, 255, 0))
        tracer_trajectoire(video_barre, self.traj_remontee, (0, 220, 255))

        faute_pieds = self.detecteur_pieds.update(
            frame=video_face,
            etat=self.etat,
            points=points,
            indice_image=self.indice_image,
            ajouter_evenement=self.add_event,
        )
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

    def _build_dashboard_sections(
        self,
        y_bassin_lisse,
        largeur_bassin_lisse,
        vitesse,
        angle_genou_lisse,
    ):
        verdict_profondeur = "-"
        if self.etat in ("remontee", "termine") and self.image_debut_remontee is not None:
            verdict_profondeur = "FAUTE" if self.faute_descente_insuffisante else "OK"

        trajectoire_txt = "-"
        if self.trajectoire_ok is True:
            trajectoire_txt = "OK"
        elif self.trajectoire_ok is False:
            trajectoire_txt = "FAUTE"

        ecart_traj_txt = "-"
        seuil_traj_txt = "-"
        if self.trajectoire_stats is not None:
            ecart_traj_txt = f"{self.trajectoire_stats['ecart_moyen']:.1f}px"
            seuil_traj_txt = f"{self.trajectoire_stats['seuil']:.1f}px"

        fautes = []
        if self.faute_descente_insuffisante:
            fautes.append("profondeur")
        if self.dip_detected:
            fautes.append("redescente")
        if self.last_foot_fault:
            fautes.append("pieds")
        if self.trajectoire_ok is False:
            fautes.append("trajectoire")

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
                    f"État barre latérale : {self.phase_barre_laterale if self.phase_barre_laterale is not None else '—'}",
                    f"Début descente : {self.image_debut_descente_barre if self.image_debut_descente_barre is not None else '—'}",
                    f"Début remontée : {self.image_debut_remontee_barre if self.image_debut_remontee_barre is not None else '—'}",
                    f"Fin remontée   : {self.image_fin_remontee_barre if self.image_fin_remontee_barre is not None else '—'}",
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
                "ANALYSE TRAJECTOIRE BARRE",
                [
                    f"Source               : {'Vue latérale indépendante' if len(self.video_paths) > 1 else 'Vue face (pas de barre)'}",
                    f"Points descente lat. : {len(self.traj_descente)}",
                    f"Points remontée lat. : {len(self.traj_remontee)}",
                    f"Verdict trajectoire  : {trajectoire_txt}",
                    f"Écart moyen          : {ecart_traj_txt}",
                    f"Seuil toléré         : {seuil_traj_txt}",
                ],
            ),
            (
                "ANALYSE PIEDS",
                [
                    f"Calibration OK       : {'Oui' if self.detecteur_pieds.calibration_faite else 'Non (en cours)'}",
                    f"État                 : {'FAUTE' if self.last_foot_fault else 'OK'}",
                    f"Compteur hors seuil  : {self.detecteur_pieds.compteur_hors_seuil}/{self.detecteur_pieds.nb_images_persistance}",
                ],
            ),
        ]
        return verdict_global, sections
