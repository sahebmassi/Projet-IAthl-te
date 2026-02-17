#!/bin/bash

# Définir le chemin du sous-script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
GEN_SCRIPT="$SCRIPT_DIR/scripts/generateRecommendations.sh"
INST_SCRIPT="$SCRIPT_DIR/scripts/installVSCExtensions.sh"

# Rendre exécutable le script si nécessaire
if [ ! -x "$GEN_SCRIPT" ]; then
    echo "Ajout des droits d'exécution à : $GEN_SCRIPT"
    sudo chmod +x "$GEN_SCRIPT"
fi

# Exécution du sous-script
"$GEN_SCRIPT"

for (( i=1; i <= $#; i++ )); do

    if [ "${!i}" == "-e" ]; then
        echo "Utilisation du flag : ${!i} -> début de l'installation des extensions"

        # Vérifier la présence de --force après -e
        next=$((i+1))
        if [ "${!next}" == "--force" ]; then
            sudo chmod +x "$INST_SCRIPT"
            "$INST_SCRIPT" --force
        else
            sudo chmod +x "$INST_SCRIPT"
            "$INST_SCRIPT"
        fi

    elif [ "${!i}" == "-i" ]; then
        echo "Création de l'environnement Conda"
        conda env create -f ./config/env.yml

    elif [ "${!i}" == "-u" ]; then
        echo "Mise à jour de l'environnement Conda"
        conda env update -f ./config/env.yml --prune
    fi
done

# Ouvrir VSCode sur MacOS
if [[ "$(uname)" == "Darwin" ]]; then
    open /Applications/Visual\ Studio\ Code.app
fi
