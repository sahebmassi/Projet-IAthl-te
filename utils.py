import numpy as np
import torch

def calculerAngle(coordonnees : list, idA : int, idB : int, idC :int) -> float:
    """Calcul l'angle en A B et C sachant que B est le point de l'angle calculer
    Args:
        coordonnees (list): les coordonnées sur lesquelles on calcul l'angle
        idA (int): indice du point A
        idB (int): indice du point B
        idC (int): indice du point C
    Returns:
        float: l'angle"""
    a = np.array(coordonnees[idA])
    b = np.array(coordonnees[idB])
    c = np.array(coordonnees[idC])

    ba = a - b
    bc = b - c

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def choixDevice() -> str:
    """Choisi l'utilisation du GPU s'il est disponible, sinon choisi le CPU
    Returns:
        string: cuda si le gpu est disponible sinon cpu"""
    if torch.cuda.is_available():
        return "cuda"
    elif  torch.backends.mps.is_available():  # Pour Apple Silicon (M1/M2/M3)
        return "mps"
    else:
        return "cpu"