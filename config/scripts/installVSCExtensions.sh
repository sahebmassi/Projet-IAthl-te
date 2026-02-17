#!/bin/bash


# Dans le cas où la pop-up de vscode n'est pas apparu,
# ce script permet d'installer les extensions
# en règle général, vscode active les extensions automatiquement
# si ce n'est pas le cas, vous devrez les activer manuellement  

########################################
### Lire et installer les extensions ###
########################################


# Obtenir le chemin absolu du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Chemin vers le fichier des extensions
EXT_FILE="$SCRIPT_DIR/extensionsVSCode.txt"

# Vérifier si le fichier des extensions existe
if [ ! -f "$EXT_FILE" ]; then
    echo "Erreur : Le fichier des extensions est introuvable à : $EXT_FILE"
    exit 1
fi

# Lire et installer chaque extension
while IFS= read -r ext; do
    if [ -n "$ext" ]; then
        echo "Installation de l'extension : $ext"
        code --install-extension "$ext" "$1"
    fi
done < "$EXT_FILE"

echo "Installation des extensions terminée."
