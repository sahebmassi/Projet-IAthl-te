"""Rendu des images et du tableau de bord.

Les analyseurs produisent des vues OpenCV annotées et des sections de texte.
Ce module convertit les images vers Qt, compose les vues en mosaïque et formate
les lignes du tableau de bord affiché dans l'interface.
"""

import os
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtGui import QImage

from .constants import COULEUR_FOND_PANNEAU_BGR, LARGEUR_PANNEAU, SEUIL_ANGLE_GENOU_PROFONDEUR


def trouver_police_ttf() -> Optional[str]:
    candidats = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for path in candidats:
        if os.path.exists(path):
            return path
    return None


def construire_polices(hauteur: int):
    chemin = trouver_police_ttf()
    if hauteur >= 900:
        taille_texte, taille_titre = 22, 28
    elif hauteur >= 720:
        taille_texte, taille_titre = 20, 26
    else:
        taille_texte, taille_titre = 18, 24

    if chemin:
        police_titre = ImageFont.truetype(chemin, taille_titre)
        police_texte = ImageFont.truetype(chemin, taille_texte)
        return police_titre, police_texte, chemin
    return ImageFont.load_default(), ImageFont.load_default(), None


def panneau_unicode(hauteur: int, lignes: List[str], titre: str, cache_polices: dict) -> np.ndarray:
    if hauteur not in cache_polices:
        cache_polices[hauteur] = construire_polices(hauteur)

    police_titre, police_texte, chemin_police = cache_polices[hauteur]

    panel_bgr = np.zeros((hauteur, LARGEUR_PANNEAU, 3), dtype=np.uint8)
    panel_bgr[:] = COULEUR_FOND_PANNEAU_BGR
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(panel_rgb)
    draw = ImageDraw.Draw(img)

    x = 16
    y = 12
    draw.text((x, y), titre, font=police_titre, fill=(255, 255, 255))
    y += 40

    draw.line((x, y + 20, LARGEUR_PANNEAU - 16, y + 20), fill=(100, 100, 100), width=2)
    y += 14

    if chemin_police is None:
        draw.text(
            (x, y),
            "Police TTF non trouvee : installe DejaVu pour les accents.",
            font=police_texte,
            fill=(255, 200, 120),
        )
        y += 28

    marge = 8
    for ligne in lignes:
        if y > hauteur - 24:
            break
        draw.text((x, y), ligne, font=police_texte, fill=(235, 235, 235))
        bbox = draw.textbbox((x, y), ligne, font=police_texte)
        y += (bbox[3] - bbox[1]) + marge

    out_rgb = np.array(img)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def composer_vues(
    vues: List[np.ndarray], panneau: Optional[np.ndarray] = None
) -> np.ndarray:
    """Resize views to a common height and concatenate them horizontally."""

    if not vues:
        return panneau if panneau is not None else np.zeros((1, 1, 3), dtype=np.uint8)

    hauteur = panneau.shape[0] if panneau is not None else vues[0].shape[0]
    vues_redim = []
    for vue in vues:
        vue_h, vue_w = vue.shape[:2]
        new_w = int(vue_w * (hauteur / vue_h))
        vues_redim.append(cv2.resize(vue, (new_w, hauteur)))

    mosaic = np.hstack(vues_redim)
    if panneau is None:
        return mosaic
    return np.hstack([mosaic, panneau])


def frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to a Qt QImage."""

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def build_dashboard_lines(
    *,
    ips_txt: str,
    indice_image: int,
    t_sec: float,
    etat: str,
    verdict_global: str,
    image_debut_descente,
    image_debut_remontee,
    image_fin_remontee,
    y_txt: str,
    largeur_txt: str,
    vitesse_txt: str,
    angle_txt: str,
    angle_min_txt: str,
    image_point_bas,
    angle_pb_txt: str,
    hanches_sous_genoux,
    angle_genou_ok,
    verdict_profondeur: str,
    dip_detected: bool,
    dip_start_frame,
    dip_amp_px,
    source_trajectoire_txt: str,
    points_descente: int,
    points_remontee: int,
    trajectoire_txt: str,
    ecart_traj_txt: str,
    seuil_traj_txt: str,
    calibration_faite: bool,
    faute_pied_avant: bool,
    compteur_hors_seuil: int,
    nb_images_persistance: int,
    journal,
) -> List[str]:
    lignes = [
        f"Cadence : {ips_txt}",
        f"Image : {indice_image}    Temps : {t_sec:.2f} s",
        f"Phase : {etat}",
        f"Verdict global : {verdict_global}",
        "",
        "=== REPERES TEMPORELS ===",
        f"Debut descente : {image_debut_descente if image_debut_descente is not None else '-'}",
        f"Debut remontee : {image_debut_remontee if image_debut_remontee is not None else '-'}",
        f"Fin remontee   : {image_fin_remontee if image_fin_remontee is not None else '-'}",
        "",
        "=== DONNEES LISSEES ===",
        f"Y bassin (px)        : {y_txt}",
        f"Largeur bassin (px)  : {largeur_txt}",
        f"Vitesse bassin (px/f): {vitesse_txt}",
        f"Angle genou (deg)    : {angle_txt}",
        f"Angle genou min obs. : {angle_min_txt}",
        "",
        "=== ANALYSE REDESCENTE (DIP) ===",
        f"Dip detecte ?        : {'Detecte' if dip_detected else 'Non detecte'}",
        f"Frame debut dip      : {dip_start_frame if dip_start_frame is not None else '-'}",
        f"Amplitude dip (px)   : {dip_amp_px:.1f}" if dip_amp_px is not None else "Amplitude dip (px)   : -",
        "",
        "=== ANALYSE PROFONDEUR ===",
        f"Image point bas      : {image_point_bas if image_point_bas is not None else '-'}",
        f"Angle genou au point bas : {angle_pb_txt}",
        f"Hanches sous genoux  : {hanches_sous_genoux if hanches_sous_genoux is not None else '-'}",
        f"Angle < {SEUIL_ANGLE_GENOU_PROFONDEUR:.0f}deg : {angle_genou_ok if angle_genou_ok is not None else '-'}",
        f"Verdict profondeur   : {verdict_profondeur}",
        "",
        "=== ANALYSE TRAJECTOIRE BARRE ===",
        f"Source               : {source_trajectoire_txt}",
        f"Points descente      : {points_descente}",
        f"Points remontee      : {points_remontee}",
        f"Verdict trajectoire  : {trajectoire_txt}",
        f"Ecart moyen          : {ecart_traj_txt}",
        f"Seuil tolere         : {seuil_traj_txt}",
        "",
        "=== ANALYSE PIEDS ===",
        f"Calibration OK       : {'Oui' if calibration_faite else 'Non (en cours)'}",
        f"Etat                 : {'FAUTE' if faute_pied_avant else 'OK'}",
        f"Compteur hors seuil  : {compteur_hors_seuil}/{nb_images_persistance}",
        "",
        "=== EVENEMENTS RECENTS ===",
    ]
    lignes.extend(list(journal))
    return lignes


def build_dashboard_lines_from_sections(
    *,
    ips_txt: str,
    indice_image: int,
    t_sec: float,
    etat: str,
    verdict_global: str,
    sections,
    journal,
) -> List[str]:
    """Build final dashboard lines from analyzer-provided sections."""

    lignes = [
        f"Cadence : {ips_txt}",
        f"Image : {indice_image}    Temps : {t_sec:.2f} s",
        f"Phase : {etat}",
        f"Verdict global : {verdict_global}",
    ]

    for titre, section_lines in sections:
        lignes.append("")
        lignes.append(f"=== {titre} ===")
        lignes.extend(section_lines)

    lignes.append("")
    lignes.append("=== ÉVÉNEMENTS RÉCENTS ===")
    lignes.extend(list(journal))
    return lignes
