"""Thread de lecture vidéo et d'orchestration des analyseurs.

Le worker lit toutes les vidéos en parallèle, avance chaque capture d'une frame
à chaque boucle, puis n'envoie à l'analyseur qu'une frame sur N selon
processing_fps. Ce sous-échantillonnage est commun aux vues, mais les indices
envoyés restent des indices de frames source. C'est pourquoi les calculs de
temps utilisent les FPS source des vidéos, pas le FPS traitement.

La vue 1 fournit fps. La vue 2, si présente, fournit lateral_fps pour les
mesures de vitesse sur le disque/barre latéral.
"""

import os
import traceback
from typing import List, Optional

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .deadlift_analyzer import DeadliftAnalyzer
from .rendering import frame_to_qimage
from .squat_analyzer import SquatAnalyzer


class VideoWorker(QThread):
    image_ready = Signal(QImage)
    dashboard_ready = Signal(str)
    status_ready = Signal(str)
    finished_cleanly = Signal()
    error_signal = Signal(str)

    def __init__(
        self,
        *,
        movement: str,
        video_paths: List[str],
        pose_model_path: str,
        barbell_model_path: Optional[str],
        disk_model_path: Optional[str],
        lateral_athlete_model_path: Optional[str],
        processing_fps: float,
        conf: float,
    ):
        super().__init__()
        self.movement = movement
        self.video_paths = video_paths
        self.pose_model_path = pose_model_path
        self.barbell_model_path = barbell_model_path
        self.disk_model_path = disk_model_path
        self.lateral_athlete_model_path = lateral_athlete_model_path
        self.processing_fps = processing_fps
        self.conf = conf
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

    def _build_analyzer(self, fps: float, frame_height: int, lateral_fps: Optional[float] = None):
        """Instantiate the analyzer matching the selected movement."""

        common_kwargs = dict(
            video_paths=self.video_paths,
            fps=fps,
            lateral_fps=lateral_fps,
            processing_fps=self.processing_fps,
            frame_height=frame_height,
            conf=self.conf,
            emit_status=self._emit_status,
            pose_model_path=self.pose_model_path,
            barbell_model_path=self.barbell_model_path,
            disk_model_path=self.disk_model_path,
            lateral_athlete_model_path=self.lateral_athlete_model_path,
        )
        if self.movement == "Squat":
            return SquatAnalyzer(**common_kwargs)
        if self.movement == "Souleve de terre":
            return DeadliftAnalyzer(**common_kwargs)
        raise ValueError(f"Mouvement non supporte: {self.movement}")

    def run(self):
        """Read videos, apply frame skipping, and emit image/dashboard updates."""

        try:
            if not self.video_paths:
                raise FileNotFoundError("Aucune video fournie.")
            for video_path in self.video_paths:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video introuvable: {video_path}")
            if not os.path.exists(self.pose_model_path):
                raise FileNotFoundError(
                    f"Modele pose introuvable: {self.pose_model_path}"
                )

            while not self._stop:
                caps = [cv2.VideoCapture(video_path) for video_path in self.video_paths]
                if not all(cap.isOpened() for cap in caps):
                    raise RuntimeError("Impossible d'ouvrir une des videos.")

                frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
                images_par_seconde = float(caps[0].get(cv2.CAP_PROP_FPS))
                if images_par_seconde <= 1e-6:
                    images_par_seconde = 30.0
                lateral_fps = images_par_seconde
                if len(caps) > 1:
                    lateral_fps = float(caps[1].get(cv2.CAP_PROP_FPS))
                    if lateral_fps <= 1e-6:
                        lateral_fps = images_par_seconde

                analyzer = self._build_analyzer(
                    images_par_seconde,
                    frame_height,
                    lateral_fps,
                )

                last_frames = [None] * len(caps)
                frame_count = 0
                while not self._stop:
                    if self._restart:
                        self._restart = False
                        break
                    if self._pause:
                        self.msleep(30)
                        continue

                    frames = []
                    for i, cap in enumerate(caps):
                        ok, frame = cap.read()
                        if ok:
                            last_frames[i] = frame.copy()
                            frames.append(frame)
                        else:
                            frames.append(last_frames[i] if last_frames[i] is not None else None)

                    if not any(f is not None for f in frames):
                        self._emit_status("Fin de toutes les videos")
                        self.finished_cleanly.emit()
                        for cap in caps:
                            cap.release()
                        return

                    frame_count += 1
                    if frame_count % analyzer.frame_stride != 0:
                        continue

                    combo = analyzer.process_frame(frames, frame_count - 1)
                    self.image_ready.emit(frame_to_qimage(combo))
                    self.dashboard_ready.emit(analyzer.last_dashboard_text)
                    self.msleep(max(1, int(1000 / analyzer.processing_fps)))

                for cap in caps:
                    cap.release()

        except Exception:
            self.error_signal.emit(traceback.format_exc())
