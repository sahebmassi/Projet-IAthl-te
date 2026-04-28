"""Base commune des analyseurs de mouvements.

Ce module contient la classe abstraite utilisée par les analyseurs Squat et
Deadlift. Elle centralise le chargement des modèles YOLO, les paramètres de
cadence, la génération du tableau de bord texte et les helpers d'affichage.

Deux notions de FPS sont volontairement séparées:
- fps: FPS source de la vue principale, généralement la vue face.
- lateral_fps: FPS source de la vue latérale. Il sert aux vitesses mesurées
  sur le disque/barre latéral quand les deux vidéos n'ont pas le même FPS.

Le FPS traitement ne remplace jamais le FPS source: il ne sert qu'à
sous-échantillonner les frames pour accélérer l'analyse.
"""

import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, List, Optional

import cv2
from ultralytics import YOLO

from .pose import choisir_personne_principale
from .rendering import build_dashboard_lines_from_sections, composer_vues


class BaseMovementAnalyzer(ABC):
    """Classe de base pour un analyseur vidéo frame par frame.

    Les sous-classes doivent implémenter process_frame(). Le worker leur donne
    des frames déjà lues pour toutes les vues. L'analyseur renvoie une image
    mosaïque, et stocke le texte du tableau de bord dans last_dashboard_text.
    """

    dashboard_title = "TABLEAU DE BORD"

    def __init__(
        self,
        *,
        video_paths: List[str],
        fps: float,
        processing_fps: float,
        frame_height: int,
        conf: float,
        emit_status: Callable[[str], None],
        pose_model_path: str,
        lateral_fps: Optional[float] = None,
        barbell_model_path: Optional[str] = None,
        disk_model_path: Optional[str] = None,
        lateral_athlete_model_path: Optional[str] = None,
        require_barbell_model: bool = False,
        require_disk_model: bool = False,
    ):
        self.video_paths = video_paths
        self.fps = fps
        self.lateral_fps = lateral_fps if lateral_fps and lateral_fps > 1e-6 else fps
        self.processing_fps = max(1.0, min(float(processing_fps), float(fps)))
        self.frame_height = frame_height
        self.conf = conf
        self.emit_status = emit_status
        self.frame_stride = max(1, int(round(fps / self.processing_fps)))
        self.cache_polices = {}
        self.journal = deque(maxlen=18)
        self.last_dashboard_text = ""

        self.pose_model_path = pose_model_path
        self.barbell_model_path = barbell_model_path
        self.disk_model_path = disk_model_path
        self.lateral_athlete_model_path = lateral_athlete_model_path

        self.pose_model = self._load_model(pose_model_path, "Modele pose", required=True)
        self.barbell_model = self._load_model(
            barbell_model_path,
            "Modele barre",
            required=require_barbell_model,
        )
        self.disk_model = self._load_model(
            disk_model_path,
            "Modele disque",
            required=require_disk_model,
        )
        self.lateral_athlete_model = self._load_model(
            lateral_athlete_model_path,
            "Modele athlete lateral",
            required=False,
        )

    def _load_model(self, path: Optional[str], label: str, required: bool):
        """Load a YOLO model or return None when the model is optional."""

        if not path:
            if required:
                raise FileNotFoundError(f"{label} manquant.")
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} introuvable: {path}")
        return YOLO(path)

    def add_event(self, frame_index: int, message: str) -> None:
        """Add a timestamped event to the dashboard journal and status bar."""

        t_sec = frame_index / self.fps if frame_index >= 0 else 0.0
        self.journal.appendleft(f"[{t_sec:5.2f}s] {message}")
        self.emit_status(message)

    def predict_pose_points(self, frame):
        """Run the pose model and return the selected main person's keypoints."""

        resultats = self.pose_model.predict(source=frame, conf=self.conf, verbose=False)
        result = resultats[0] if resultats else None
        return choisir_personne_principale(result)

    def compose_output(
        self,
        vues_annotees,
        *,
        ips_txt: str,
        indice_image: int,
        t_sec: float,
        etat: str,
        verdict_global: str,
        sections,
    ):
        """Build dashboard text and return the visual output image.

        The UI displays the dashboard in a QTextEdit, not inside the OpenCV
        mosaic. The returned image therefore contains only the annotated views.
        """

        lignes = build_dashboard_lines_from_sections(
            ips_txt=ips_txt,
            indice_image=indice_image,
            t_sec=t_sec,
            etat=etat,
            verdict_global=verdict_global,
            sections=sections,
            journal=self.journal,
        )
        self.last_dashboard_text = "\n".join([self.dashboard_title, ""] + lignes)
        return composer_vues(vues_annotees)

    def format_fps_text(self) -> str:
        """Format source and processing FPS for the dashboard."""

        return f"{self.fps:.2f} ips source | {self.processing_fps:.2f} ips traitement"

    @staticmethod
    def _draw_overlay_header(image, title: str, subtitle: str, color) -> None:
        """Draw a small title/status overlay at the top-left of a video view."""

        cv2.putText(
            image,
            title,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
        cv2.putText(
            image,
            subtitle,
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
        )

    @abstractmethod
    def process_frame(self, frames: List, source_frame_index: int) -> "cv2.typing.MatLike":
        raise NotImplementedError
