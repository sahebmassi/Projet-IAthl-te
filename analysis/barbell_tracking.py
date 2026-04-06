import os
import unicodedata
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import (
    BARBELL_MAX_JUMP_RATIO,
    NB_IMAGES_CONSEC_BARRE_PHASE,
    SEUIL_VITESSE_BARRE_LATERALE_PX,
    TRAJECTOIRE_BIN_COUNT,
    TRAJECTOIRE_ECART_RATIO,
)

BarbellPoint = Tuple[int, int]
BarbellBox = Tuple[int, int, int, int]

_DISK_ALIASES = (
    "disque",
    "dique",
    "disc",
    "plate",
    "weight_plate",
    "weight plate",
)
_BAR_ALIASES = ("barbell", "barre", "bar")


@dataclass
class DetectionSignal:
    center: Optional[BarbellPoint] = None
    box: Optional[BarbellBox] = None
    confidence: float = 0.0
    class_name: Optional[str] = None
    source: Optional[str] = None


def _normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).strip().lower())
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_value.replace("-", " ").replace("_", " ").strip()


def resolve_barbell_target_class(
    model_path: str,
    model_names: Dict[int, str],
    preferred_kind: str = "auto",
) -> Optional[str]:
    if not model_names:
        return None

    normalized_to_original = {
        _normalize_label(class_name): class_name for class_name in model_names.values()
    }
    basename = os.path.basename(model_path).lower()

    if preferred_kind == "disk":
        ordered_aliases = list(_DISK_ALIASES) + list(_BAR_ALIASES)
    elif preferred_kind == "bar":
        ordered_aliases = list(_BAR_ALIASES) + list(_DISK_ALIASES)
    elif basename in {"best.pt", "disque.pt"}:
        ordered_aliases = list(_DISK_ALIASES) + list(_BAR_ALIASES)
    elif basename == "barre.pt":
        ordered_aliases = list(_BAR_ALIASES) + list(_DISK_ALIASES)
    else:
        ordered_aliases = list(_BAR_ALIASES) + list(_DISK_ALIASES)

    for alias in ordered_aliases:
        match = normalized_to_original.get(_normalize_label(alias))
        if match is not None:
            return match

    if len(model_names) == 1:
        return next(iter(model_names.values()))

    return None


def choisir_meilleure_box_barbell(
    result,
    model_names: Dict[int, str],
    target_class: Optional[str],
):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None

    target_normalized = _normalize_label(target_class) if target_class else None
    if target_normalized is None and len(model_names) > 1:
        return None

    best_box = None
    best_conf = -1.0

    for box in result.boxes:
        try:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
        except Exception:
            continue

        class_name = model_names.get(cls_id, str(cls_id))
        if target_normalized and _normalize_label(class_name) != target_normalized:
            continue

        if conf > best_conf:
            best_conf = conf
            best_box = box

    return best_box


def centre_barbell_filtre(best_box, historique_centres: Deque[BarbellPoint]):
    if best_box is None:
        return None

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    largeur = max(1, x2 - x1)
    hauteur = max(1, y2 - y1)
    max_jump = max(10.0, BARBELL_MAX_JUMP_RATIO * max(largeur, hauteur))

    if historique_centres:
        px, py = historique_centres[-1]
        if abs(cx - px) > max_jump or abs(cy - py) > max_jump:
            return None

    historique_centres.append((cx, cy))
    xs = [point[0] for point in historique_centres]
    ys = [point[1] for point in historique_centres]
    sx = int(round(float(np.median(xs))))
    sy = int(round(float(np.median(ys))))
    return (sx, sy), (x1, y1, x2, y2)


def tracked_detection_signal(
    result,
    model_names: Dict[int, str],
    target_class: Optional[str],
    history: Deque[BarbellPoint],
    source: str,
) -> DetectionSignal:
    best_box = choisir_meilleure_box_barbell(result, model_names, target_class)
    if best_box is None:
        return DetectionSignal(source=source)

    filtered = centre_barbell_filtre(best_box, history)
    if filtered is None:
        return DetectionSignal(source=source)

    center, box = filtered
    try:
        confidence = float(best_box.conf[0])
        cls_id = int(best_box.cls[0])
        class_name = model_names.get(cls_id, str(cls_id))
    except Exception:
        confidence = 0.0
        class_name = target_class

    return DetectionSignal(
        center=center,
        box=box,
        confidence=confidence,
        class_name=class_name,
        source=source,
    )


def fuse_detection_signals(
    preferred_signal: DetectionSignal,
    fallback_signal: DetectionSignal,
) -> DetectionSignal:
    if preferred_signal.center is None:
        return fallback_signal
    if fallback_signal.center is None:
        return preferred_signal

    px, py = preferred_signal.center
    fx, fy = fallback_signal.center
    preferred_box = preferred_signal.box or (px, py, px, py)
    fallback_box = fallback_signal.box or (fx, fy, fx, fy)
    preferred_size = max(
        1,
        preferred_box[2] - preferred_box[0],
        preferred_box[3] - preferred_box[1],
    )
    fallback_size = max(
        1,
        fallback_box[2] - fallback_box[0],
        fallback_box[3] - fallback_box[1],
    )
    max_gap = max(12.0, 0.75 * float(max(preferred_size, fallback_size)))
    distance = float(np.hypot(px - fx, py - fy))

    if distance > max_gap:
        return preferred_signal

    weight_preferred = max(0.05, preferred_signal.confidence)
    weight_fallback = max(0.05, fallback_signal.confidence)
    total_weight = weight_preferred + weight_fallback
    cx = int(round((px * weight_preferred + fx * weight_fallback) / total_weight))
    cy = int(round((py * weight_preferred + fy * weight_fallback) / total_weight))
    box = (
        preferred_signal.box
        if preferred_signal.confidence >= fallback_signal.confidence
        else fallback_signal.box
    )
    return DetectionSignal(
        center=(cx, cy),
        box=box,
        confidence=max(preferred_signal.confidence, fallback_signal.confidence),
        class_name=preferred_signal.class_name or fallback_signal.class_name,
        source=f"{preferred_signal.source}+{fallback_signal.source}",
    )


def draw_detection_signal(
    image: np.ndarray,
    signal: DetectionSignal,
    color_box: Tuple[int, int, int],
    color_center: Tuple[int, int, int],
    label: Optional[str] = None,
) -> None:
    if signal.box is not None:
        x1, y1, x2, y2 = signal.box
        cv2.rectangle(image, (x1, y1), (x2, y2), color_box, 2)
        if label:
            text = label
            if signal.class_name:
                text = f"{label}: {signal.class_name}"
            cv2.putText(
                image,
                text,
                (x1, max(14, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_box,
                1,
            )
    if signal.center is not None:
        cv2.circle(image, signal.center, 5, color_center, -1)


def tracer_trajectoire(
    image: np.ndarray,
    points: List[BarbellPoint],
    couleur: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    for i in range(1, len(points)):
        cv2.line(image, points[i - 1], points[i], couleur, 2)


def comparer_trajectoires(
    desc_points: List[BarbellPoint],
    rem_points: List[BarbellPoint],
    largeur_ref: Optional[float],
):
    if len(desc_points) < 4 or len(rem_points) < 4 or not largeur_ref:
        return None, None

    desc_sorted = sorted(desc_points, key=lambda point: point[1])
    rem_sorted = sorted(rem_points, key=lambda point: point[1])
    y_min = max(desc_sorted[0][1], rem_sorted[0][1])
    y_max = min(desc_sorted[-1][1], rem_sorted[-1][1])
    if y_max - y_min < 5:
        return None, None

    ys_desc = [point[1] for point in desc_sorted]
    ys_rem = [point[1] for point in rem_sorted]
    bins = np.linspace(y_min, y_max, TRAJECTOIRE_BIN_COUNT)
    ecarts = []

    def interp_x(points_sorted, ys_sorted, y_query):
        idx = bisect_left(ys_sorted, y_query)
        if idx <= 0:
            return float(points_sorted[0][0])
        if idx >= len(points_sorted):
            return float(points_sorted[-1][0])

        x0, y0 = points_sorted[idx - 1]
        x1, y1 = points_sorted[idx]
        if y1 == y0:
            return float(x0)

        alpha = (y_query - y0) / (y1 - y0)
        return float(x0 + alpha * (x1 - x0))

    for y_query in bins:
        x_desc = interp_x(desc_sorted, ys_desc, float(y_query))
        x_rem = interp_x(rem_sorted, ys_rem, float(y_query))
        ecarts.append(abs(x_desc - x_rem))

    ecart_moyen = float(np.mean(ecarts)) if ecarts else None
    seuil = float(TRAJECTOIRE_ECART_RATIO * largeur_ref)
    ok = ecart_moyen is not None and ecart_moyen <= seuil
    return ok, {"ecart_moyen": ecart_moyen, "seuil": seuil, "nb_bins": len(ecarts)}


def phase_barre_depuis_vue_face(
    indice_image: int,
    image_debut_descente: Optional[int],
    image_debut_remontee: Optional[int],
    image_fin_remontee: Optional[int],
) -> Optional[str]:
    if image_debut_descente is None or indice_image < image_debut_descente:
        return None
    if image_debut_remontee is None or indice_image < image_debut_remontee:
        return "descente"
    if image_fin_remontee is None or indice_image <= image_fin_remontee:
        return "remontee"
    return None


def phase_barre_depuis_vue_laterale(
    y_barre: float,
    y_barre_prev: Optional[float],
    phase_actuelle: Optional[str],
    compteur_descente: int,
    compteur_remontee: int,
) -> Tuple[Optional[str], int, int]:
    if y_barre_prev is None:
        return phase_actuelle, compteur_descente, compteur_remontee

    dy = y_barre - y_barre_prev

    if dy >= SEUIL_VITESSE_BARRE_LATERALE_PX:
        compteur_descente += 1
        compteur_remontee = 0
    elif dy <= -SEUIL_VITESSE_BARRE_LATERALE_PX:
        compteur_remontee += 1
        compteur_descente = 0
    else:
        compteur_descente = max(0, compteur_descente - 1)
        compteur_remontee = max(0, compteur_remontee - 1)

    if compteur_descente >= NB_IMAGES_CONSEC_BARRE_PHASE:
        phase_actuelle = "descente"
    elif compteur_remontee >= NB_IMAGES_CONSEC_BARRE_PHASE:
        phase_actuelle = "remontee"

    return phase_actuelle, compteur_descente, compteur_remontee
