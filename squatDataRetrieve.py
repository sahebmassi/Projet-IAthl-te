from enumIndice import *
from utils import *


genoux = IndiceYolo.GENOU_GAUCHE.value, IndiceYolo.GENOU_DROIT.value
hanches = IndiceYolo.HANCHE_GAUCHE.value, IndiceYolo.HANCHE_DROITE.value
epaules = IndiceYolo.EPAULE_GAUCHE.value, IndiceYolo.EPAULE_DROITE.value
talons = IndiceYolo.TALON_GAUCHE.value, IndiceYolo.TALON_DROIT.value

listeIndiceSquat = [epaules[0], hanches[0], genoux[0], talons[0], epaules[1], hanches[1], genoux[1], talons[1]]



def traitementSquat(toutesCoordonnees : list) -> tuple[list, list]:
    """Ressort les coordonnées pour vérifier le mouvement
    Args:
        toutesCoordonnees (list): les coordonées depuis le début du mouvement
    Returns:
        tuple[list, list]: coord et angles à afficher
    """
    articulations = toutesCoordonnees[-1][0]
    if articulations != []:
        subjectArticulations = articulations[0]
        angleGauche = calculerAngle(subjectArticulations, hanches[0], genoux[0], talons[0])
        angleDroit = calculerAngle(subjectArticulations, hanches[1], genoux[1], talons[1])
        listArt = [subjectArticulations[elem] for elem in listeIndiceSquat]
        return ([listArt], [angleGauche, angleDroit])
    else:
        return ([], [0, 0])