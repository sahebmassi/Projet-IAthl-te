"""Constantes de configuration du projet.

Les seuils sont regroupés ici pour éviter de les disperser dans les analyseurs.
Les chemins de modèles par défaut sont résolus relativement au dossier du
projet, ce qui permet de lancer l'application depuis la racine sans config
supplémentaire.
"""

from pathlib import Path


DEBUG_PIEDS = True
DEBUG_PIEDS_EVERY = 5
PROCESSING_FPS_DEFAULT = 15


def _resolve_default_model_path(filename: str) -> str:
    package_root = Path(__file__).resolve().parent.parent
    legacy_root = package_root / "Projet-IAthl-te"
    candidates = [
        package_root / filename,
        legacy_root / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return filename


def get_available_pt_models() -> list[str]:
    """Récupère tous les fichiers .pt disponibles dans le répertoire courant,
    exclusion des modèles lift_lateral qui ont une interface dédiée."""
    cwd = Path.cwd()
    pt_files = sorted([f.name for f in cwd.glob("*.pt")])
    # Exclure les modèles lift_lateral
    pt_files = [f for f in pt_files if "lift_lateral" not in f]
    return pt_files if pt_files else ["Aucun modèle trouvé"]


def get_lateral_athlete_models() -> list[str]:
    """Récupère les modèles lift_lateral disponibles pour l'athlète en vue latérale."""
    cwd = Path.cwd()
    pt_files = sorted([f.name for f in cwd.glob("*.pt") if "lift_lateral" in f.name])
    return pt_files if pt_files else ["Aucun modèle trouvé"]


POSE_MODEL_DEFAULT = _resolve_default_model_path("yolov8n-pose.pt")
SQUAT_BARBELL_MODEL_DEFAULT = _resolve_default_model_path("best.pt")
DEADLIFT_BAR_MODEL_DEFAULT = _resolve_default_model_path("barre.pt")
DEADLIFT_FACE_BAR_MODEL_DEFAULT = _resolve_default_model_path("barre_face.pt")
DISK_MODEL_DEFAULT = _resolve_default_model_path("disque.pt")
BARBELL_MODEL_DEFAULT = SQUAT_BARBELL_MODEL_DEFAULT
LATERAL_ATHLETE_MODEL_DEFAULT = _resolve_default_model_path("lift_lateral_finetuned.pt")

FOOT_CALIBRATION_FRAMES = 30
FOOT_FLOW_ACCUMULATION_WINDOW = 10
FOOT_DISPLACEMENT_THRESHOLD = 90.0
FOOT_PERSISTANCE_FRAMES = 25
FOOT_GRACE_PERIOD_FRAMES = 100
FOOT_VERTICAL_WEIGHT = 0.10
SQUAT_FOOT_DISPLACEMENT_RATIO_HIPWIDTH = 0.50

SEUIL_VITESSE_RATIO_LARGEUR_BASSIN = 0.008
FENETRE_LISSAGE = 5
NB_IMAGES_CONSEC_DESCENTE = 4
NB_IMAGES_CONSEC_REMONTEE = 4
DUREE_MIN_DESCENTE_A_60_IPS = 6

SEUIL_VERROUILLAGE_GENOU = 173.0
DUREE_VERROUILLAGE_SEC = 0.12
BANDE_HAUT_RATIO_LARGEUR_BASSIN = 0.035
DUREE_HAUT_SEC = 0.12
SEUIL_STABILITE_RATIO_LARGEUR_BASSIN = 0.006
DELAI_MIN_AVANT_FIN_SEC = 0.25

SEUIL_ANGLE_GENOU_PROFONDEUR = 90.0

K_MIN_DIP_SEC = 0.06
X_IGNORE_AFTER_ASCENT_SEC = 0.05
A_MIN_DIP_RATIO_OF_HIPWIDTH = 0.015

CALIBRATION_FRAMES = 30
RAYON_TOLERANCE_RATIO = 0.10
SEUIL_MIN_PIXELS = 6.0
NB_CONSECUTIF_PIED_AVANT = 6

LARGEUR_PANNEAU = 680
COULEUR_FOND_PANNEAU_BGR = (28, 28, 28)

BARBELL_CONF_THRES = 0.20
BARBELL_MAX_JUMP_RATIO = 0.35
BARBELL_SMOOTH_WINDOW = 5
TRAJECTOIRE_BIN_COUNT = 12
TRAJECTOIRE_ECART_RATIO = 0.12
SEUIL_VITESSE_BARRE_LATERALE_PX = 2.0
NB_IMAGES_CONSEC_BARRE_PHASE = 3

# Constantes pour le squelette latéral
LATERAL_KPT_EDGES_5 = [(0, 1), (1, 2), (2, 3), (3, 4)]

HANCHE_G, HANCHE_D = 11, 12
EPAULE_G, EPAULE_D = 5, 6
GENOU_G, GENOU_D = 13, 14
CHEVILLE_G, CHEVILLE_D = 15, 16

DEADLIFT_SETUP_FRAMES = 20
DEADLIFT_START_CONSEC_FRAMES = 4
DEADLIFT_START_DISPLACEMENT_RATIO_HIPWIDTH = 0.06
DEADLIFT_START_DISPLACEMENT_MIN_PX = 6.0
DEADLIFT_START_VELOCITY_RATIO_HIPWIDTH = 0.012
DEADLIFT_START_VELOCITY_MIN_PX = 2.0
DEADLIFT_TOP_MIN_ASCENT_RATIO_HIPWIDTH = 0.22
DEADLIFT_TOP_MIN_ASCENT_MIN_PX = 18.0
DEADLIFT_TOP_CANDIDATE_FRAMES = 4
DEADLIFT_TOP_HOLD_FRAMES = 6
DEADLIFT_TOP_STABILITY_RATIO_HIPWIDTH = 0.007
DEADLIFT_TOP_STABILITY_MIN_PX = 2.0
DEADLIFT_TOP_PLATEAU_RATIO_HIPWIDTH = 0.028
DEADLIFT_TOP_PLATEAU_MIN_PX = 7.0
DEADLIFT_REDESCENT_CONSEC_FRAMES = 3
DEADLIFT_REDESCENT_RATIO_HIPWIDTH = 0.03
DEADLIFT_REDESCENT_MIN_PX = 8.0
DEADLIFT_LOCKOUT_KNEE_ANGLE = 172.0
DEADLIFT_FOOT_DISPLACEMENT_RATIO_HIPWIDTH = 0.50

SQUELETTE_COCO17 = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
