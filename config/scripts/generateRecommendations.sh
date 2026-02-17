#!/bin/bash

# Détermine le chemin absolu du répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Définir les chemins des fichiers
EXT_FILE="$SCRIPT_DIR/extensionsVSCode.txt"
WORKSPACE_FILE="$SCRIPT_DIR/../.vscode/extensions.json"

# Créer le dossier .vscode s'il n'existe pas déjà
if [ ! -d "$SCRIPT_DIR/../.vscode" ]; then
    mkdir "$SCRIPT_DIR/../.vscode"
fi

# Créer le fichier extensions.json
touch "$WORKSPACE_FILE"

echo "Génération des recommandations dans : $WORKSPACE_FILE"

# Vérifier si le fichier des extensions existe
if [ ! -f "$EXT_FILE" ]; then
    echo "Erreur : Le fichier $EXT_FILE est introuvable."
    exit 1
fi

# Début du fichier JSON
echo "{" > "$WORKSPACE_FILE"
    echo "    \"recommendations\": [" >> "$WORKSPACE_FILE"

    # Lire les extensions et les ajouter
    while IFS= read -r ext; do
        echo "        \"$ext\"," >> "$WORKSPACE_FILE"
    done < "$EXT_FILE"

    # Supprimer la dernière virgule (compatible macOS et Linux)
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' -e '$ s/,$//' "$WORKSPACE_FILE"   # macOS
    else
        sed -i -e '$ s/,$//' "$WORKSPACE_FILE"      # Linux
    fi

    # Fin du fichier JSON
    echo "    ]" >> "$WORKSPACE_FILE"
echo "}" >> "$WORKSPACE_FILE"

echo "Fichier $WORKSPACE_FILE généré avec les recommandations d'extensions."
