import os
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

    def _build_analyzer(self, fps: float, frame_height: int):
        common_kwargs = dict(
            video_paths=self.video_paths,
            fps=fps,
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

                analyzer = self._build_analyzer(images_par_seconde, frame_height)

                frame_count = 0
                while not self._stop:
                    if self._restart:
                        self._restart = False
                        break
                    if self._pause:
                        self.msleep(30)
                        continue

                    frames = []
                    ok_global = True
                    for cap in caps:
                        ok, frame = cap.read()
                        if not ok:
                            ok_global = False
                            break
                        frames.append(frame)

                    if not ok_global:
                        self._emit_status("Fin de la video")
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

        except Exception as exc:
            self.error_signal.emit(str(exc))
