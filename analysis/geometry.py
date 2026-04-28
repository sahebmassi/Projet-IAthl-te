import math
"""Calculs géométriques sur les keypoints de pose.

Les analyseurs reçoivent les keypoints YOLO au format COCO. Ce module fournit
les mesures réutilisables: angles de genoux, hauteur du bassin/épaules, largeur
du bassin et positions utiles pour juger profondeur/verrouillage.
"""

from typing import List, Optional, Tuple

from .constants import (
    CHEVILLE_D,
    CHEVILLE_G,
    EPAULE_D,
    EPAULE_G,
    GENOU_D,
    GENOU_G,
    HANCHE_D,
    HANCHE_G,
)

Point = Tuple[float, float]
Points = List[Point]


def limiter(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_degres(a: Point, b: Point, c: Point) -> Optional[float]:
    """Return the ABC angle in degrees, or None when a point is missing."""

    ax, ay = a
    bx, by = b
    cx, cy = c
    if (ax == 0 and ay == 0) or (bx == 0 and by == 0) or (cx == 0 and cy == 0):
        return None

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    norm_ba = math.hypot(ba[0], ba[1])
    norm_bc = math.hypot(bc[0], bc[1])
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosang = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cosang = limiter(cosang, -1.0, 1.0)
    return float(math.degrees(math.acos(cosang)))


def y_bassin_et_largeur(points: Points) -> Tuple[Optional[float], Optional[float]]:
    """Return average hip Y and hip width from COCO hip keypoints."""

    xg, yg = points[HANCHE_G]
    xd, yd = points[HANCHE_D]
    if (xg == 0 and yg == 0) or (xd == 0 and yd == 0):
        return None, None

    y_milieu = 0.5 * (yg + yd)
    largeur = math.hypot(xg - xd, yg - yd)
    if largeur < 1.0:
        largeur = None
    return float(y_milieu), float(largeur) if largeur is not None else None


def angle_moyen_genoux(points: Points) -> Optional[float]:
    """Return the average of visible left/right knee angles."""

    g = angle_degres(points[HANCHE_G], points[GENOU_G], points[CHEVILLE_G])
    d = angle_degres(points[HANCHE_D], points[GENOU_D], points[CHEVILLE_D])
    vals = [v for v in (g, d) if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def y_moyenne_genoux(points: Points) -> Optional[float]:
    """Return average knee Y for squat depth comparison."""

    gx, gy = points[GENOU_G]
    dx, dy = points[GENOU_D]
    if (gx == 0 and gy == 0) or (dx == 0 and dy == 0):
        return None
    return float(0.5 * (gy + dy))


def angles_genoux(points: Points) -> Tuple[Optional[float], Optional[float]]:
    """Return left and right knee angles independently."""

    gauche = angle_degres(points[HANCHE_G], points[GENOU_G], points[CHEVILLE_G])
    droite = angle_degres(points[HANCHE_D], points[GENOU_D], points[CHEVILLE_D])
    return gauche, droite


def angle_genou_min_visible(points: Points) -> Optional[float]:
    """Return the smallest visible knee angle, useful for lockout checks."""

    gauche, droite = angles_genoux(points)
    visibles = [angle for angle in (gauche, droite) if angle is not None]
    if not visibles:
        return None
    return float(min(visibles))


def y_epaules_et_largeur(points: Points) -> Tuple[Optional[float], Optional[float]]:
    """Return average shoulder Y and shoulder width."""

    xg, yg = points[EPAULE_G]
    xd, yd = points[EPAULE_D]
    if (xg == 0 and yg == 0) or (xd == 0 and yd == 0):
        return None, None

    y_milieu = 0.5 * (yg + yd)
    largeur = math.hypot(xg - xd, yg - yd)
    if largeur < 1.0:
        largeur = None
    return float(y_milieu), float(largeur) if largeur is not None else None
