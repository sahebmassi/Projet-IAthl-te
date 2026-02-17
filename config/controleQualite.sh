#!/bin/bash

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null
then
    echo "Python n'est pas installé. Veuillez installer Python avant d'exécuter ce script."
    exit 1
fi

# Vérifier si le fichier controleQualitePython.py existe
if [ ! -f "./controleQualitePython.py" ]; then
    echo "Le fichier controleQualitePython.py n'existe pas dans le répertoire courant."
    exit 1
fi

# Lancer le script Python
echo "Lancement du script Python controleQualitePython.py..."
python3 ./controleQualitePython.py

# Vérifier le statut de l'exécution du script Python
if [ $? -eq 0 ]; then
    echo "Le script s'est exécuté avec succès."
else
    echo "Le script a échoué."
    exit 1
fi
