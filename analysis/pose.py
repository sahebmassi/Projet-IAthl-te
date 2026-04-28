from typing import Optional

"""Helpers de pose.

Ce module sélectionne la personne principale dans une sortie YOLO pose et
dessine les squelettes utilisés par les vues face et latérale. Il ne contient
pas de logique de jugement sportif: uniquement extraction et rendu des points.
"""

import cv2

from .constants import CHEVILLE_D, CHEVILLE_G, LATERAL_KPT_EDGES_5, SQUELETTE_COCO17
from .geometry import Points


def dessiner_squelette(image, points: Points) -> None:
    """Draw COCO keypoints and skeleton for the face/main view."""

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


def dessiner_squelette_lateral(image, points: Points) -> None:
    """Draw a compact lateral skeleton for the lateral athlete model."""

    """Dessine le squelette pour les vues latérales avec des couleurs distinctes."""
    if not points:
        return

    for idx, (x, y) in enumerate(points):
        if x is None or y is None or (x == 0 and y == 0):
            continue
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 255), -1)  # Cyan
        cv2.putText(
            image, str(idx), (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )

    if len(points) == 5:
        edges = LATERAL_KPT_EDGES_5
    else:
        edges = [(i, i + 1) for i in range(len(points) - 1)]

    for a, b in edges:
        if a >= len(points) or b >= len(points):
            continue
        xa, ya = points[a]
        xb, yb = points[b]
        if (xa == 0 and ya == 0) or (xb == 0 and yb == 0):
            continue
        cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)  # Orange


def choisir_personne_principale(result) -> Optional[Points]:
    """Select the largest detected person and return their keypoints."""

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
