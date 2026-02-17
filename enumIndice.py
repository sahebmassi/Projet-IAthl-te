from enum import Enum

class IndiceMediaPipe(Enum):
    """correspondance entre les indice selon mediaPipe et un représentation plus compréhensible
    Args:
        Enum (str): l'endroit voulu"""
    NEZ = 0
    OEIL_GAUCHE_INTERIEUR = 1
    OEIL_GAUCHE = 2
    OEIL_GAUCHE_EXTERIEUR = 3
    OEIL_DROIT_INTERIEUR = 4
    OEIL_DROIT = 5
    OEIL_DROIT_EXTERIEUR = 6
    OREILLE_GAUCHE = 7
    OREILLE_DROITE = 8
    BOUCHE_GAUCHE = 9
    BOUCHE_DROITE = 10
    EPAULE_GAUCHE = 11
    EPAULE_DROITE = 12
    COUDE_GAUCHE = 13
    COUDE_DROIT = 14
    POIGNET_GAUCHE = 15
    POIGNET_DROIT = 16
    AURICULAIRE_GAUCHE = 17
    AURICULAIRE_DROIT = 18
    INDEX_GAUCHE = 19
    INDEX_DROIT = 20
    POUCE_GAUCHE = 21
    POUCE_DROIT = 22
    HANCHE_GAUCHE = 23
    HANCHE_DROITE = 24
    GENOU_GAUCHE = 25
    GENOU_DROIT = 26
    CHEVILLE_GAUCHE = 27
    CHEVILLE_DROITE = 28
    TALON_GAUCHE = 29
    TALON_DROIT = 30
    AVANT_PIED_GAUCHE = 31
    AVANT_PIED_DROIT = 32

class IndiceYolo(Enum):
    """correspondance entre les indice selon yolo et un représentation plus compréhensible
    Args:
        Enum (str): l'endroit voulu"""
    NEZ = 0
    OEIL_GAUCHE = 1
    OEIL_DROIT = 2
    OREILLE_GAUCHE = 3
    OREILLE_DROITE = 4
    EPAULE_GAUCHE = 5
    EPAULE_DROITE = 6
    COUDE_GAUCHE = 7
    COUDE_DROIT = 8
    POIGNET_GAUCHE = 9
    POIGNET_DROIT = 10
    HANCHE_GAUCHE = 11
    HANCHE_DROITE = 12
    GENOU_GAUCHE = 13
    GENOU_DROIT = 14
    TALON_GAUCHE = 15
    TALON_DROIT = 16

class IndicePointDeVue(Enum):
    """correspondance entre un point de vu et un entier pour faciliter l'utilisation
    Args:
        Enum (_type_): le point de vue voulu"""
    GAUCHE = 0
    CENTRE = 1
    DROITE = 2

class IndiceMouvement(Enum):
    """correspondance entre un mouvement et un entier pour faciliter l'utilisation
    Args:
        Enum (_type_): le mouvement voulu"""
    SQUAT = 0
    DEADLIFT = 1
    BENCH = 2