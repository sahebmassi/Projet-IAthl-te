#!/bin/bash

# Détection du système
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Système détecté : macOS"
    # Commande spécifique pour macOS
    brew install cloc 
elif [[ -f /etc/os-release ]]; then
    source /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        echo "Système détecté : Ubuntu"
        # Commande spécifique pour Ubuntu
        sudo apt install cloc  
    else
        echo "Système non pris en charge."
        exit 1
    fi
else
    echo "Système non pris en charge."
    exit 1
fi
