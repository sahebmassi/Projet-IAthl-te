### Installation du projet

./config/projectInstallation.sh pour installé les dépendances


### Script qui crée un json avec les extensions

Lancé le script en sudo -> il lance d'autre d'autres scripts et utilise chmod +x

-e les installes
-e --force force l'installation

-i fait l'install de conda
-u fait son upgrade

# Donner le fichier à root
sudo chown root le_script.sh

# mettre le bit setuid
sudo chmod +s le_script.sh

### Script controle qualite du code

controleQualite.sh
