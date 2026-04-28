Projet IAthlete - Analyse powerlifting
======================================

Objectif
--------
Cette application PySide6 analyse des videos de powerlifting avec YOLO:
- Squat: vue face + option vue laterale pour disque/barre.
- Deadlift: vue face/corps + option vue laterale pour disque.

L'interface affiche la video annotee et un tableau de bord texte avec les phases,
les fautes et les mesures calculees.


Installation
------------
Prerequis:
- Windows recommande pour cette version du projet.
- Python 3.10 ou 3.11 recommande.
- Les fichiers modele .pt doivent rester a la racine du projet:
  - yolov8n-pose.pt
  - best_centre_point.pt ou best.pt selon le modele de barre/disque disponible
  - disque.pt
  - barre_face.pt
  - lift_lateral_finetuned.pt, optionnel pour le squelette lateral deadlift

Depuis la racine du projet:

1. Creer un environnement virtuel:

   python -m venv .venv

2. Activer l'environnement:

   .venv\Scripts\activate

3. Installer les dependances:

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

4. Lancer l'application:

   python app.py

Alternative:

   python -m analysis

Si la commande python ouvre le Microsoft Store, installer Python depuis
python.org ou utiliser l'interpreteur deja present dans .venv:

   .venv\Scripts\python.exe app.py


Dependances principales
-----------------------
Le fichier requirements.txt contient:
- PySide6: interface graphique.
- opencv-python: lecture video et dessin.
- numpy: calcul numerique.
- ultralytics: modeles YOLO.
- Pillow: rendu unicode du tableau de bord.

Ultralytics installe aussi PyTorch si necessaire. Pour une machine avec GPU
NVIDIA, il peut etre utile d'installer une version PyTorch CUDA adaptee depuis
la documentation officielle de PyTorch avant ultralytics.


Utilisation de l'interface
--------------------------
1. Choisir le type de mouvement:
   - Squat
   - Souleve de terre
   - Developpe couche: pas encore implemente

2. Choisir le nombre de vues:
   - 1 vue: analyse de la vue principale uniquement.
   - 2 vues: vue principale + vue laterale.
   - 3 vues: champ prevu, mais la logique principale utilise surtout les deux
     premieres vues aujourd'hui.

3. Choisir les modeles:
   - Modele pose: generalement yolov8n-pose.pt.
   - Squat: modele barre/disque selon le fichier disponible.
   - Deadlift: disque.pt est selectionne par defaut pour la vue laterale.
   - Modele athlete lateral: optionnel, utilise pour dessiner un squelette
     lateral en deadlift si disponible.

4. Choisir les videos avec "Parcourir".
   - Squat:
     - Vue 1: face.
     - Vue 2: laterale disque/barre.
   - Deadlift:
     - Vue 1: face/corps.
     - Vue 2: laterale disque.

5. Regler "FPS traitement".
   Ce parametre ne change pas le FPS reel des videos. Il indique combien
   d'images par seconde sont analysees pour accelerer le traitement.
   Exemple: video 60 FPS + traitement 15 FPS = analyse d'environ 1 frame sur 4.
   Les calculs de temps utilisent les FPS source lus dans les videos.

6. Cliquer sur "Lancer".


Architecture du code
--------------------
app.py
  Point d'entree principal.

analysis/app.py
  Initialise QApplication et ouvre MainWindow.

analysis/ui.py
  Interface graphique: choix des videos, modeles, FPS traitement, boutons,
  affichage video et tableau de bord.

analysis/video_worker.py
  Thread de lecture video. Lit toutes les vues, applique le sous-echantillonnage
  et envoie les frames a l'analyseur choisi. Lit aussi le FPS de la vue laterale
  pour les calculs de vitesse disque.

analysis/base_analyzer.py
  Base commune: chargement YOLO, FPS, journal d'evenements, composition du
  tableau de bord.

analysis/squat_analyzer.py
  Analyse squat: phases face, profondeur, redescente/dip, pieds, trajectoire
  laterale et vitesses disque.

analysis/deadlift_analyzer.py
  Analyse deadlift: signal corps, signal disque lateral, redescente, pieds,
  trajectoire et vitesse de remontee.

analysis/barbell_tracking.py
  Fonctions de detection, filtrage et phase laterale disque/barre.

analysis/foot_detector.py
  Detection de deplacement des pieds.

analysis/geometry.py
  Calculs d'angles et mesures sur keypoints.

analysis/pose.py
  Selection de personne principale et dessin des squelettes.

analysis/rendering.py
  Conversion OpenCV -> Qt et generation du tableau de bord texte.

analysis/constants.py
  Chemins de modeles et seuils de detection.


Etat actuel - Squat
-------------------
Ce qui marche:
- Detection des phases depuis la vue face:
  - attente
  - descente
  - remontee
  - termine
- Analyse profondeur:
  - hanches sous genoux si possible
  - angle genou minimum observe
- Detection redescente/dip apres debut de remontee.
- Detection deplacement des pieds avec seuil dynamique.
- Suivi disque/barre en vue laterale.
- Trajectoire laterale:
  - vert: descente
  - jaune: remontee
- La vue face ne doit plus effacer ni piloter la trajectoire laterale.
- Calcul vitesse moyenne:
  - descente: point haut vert -> point bas vert
  - remontee: point bas jaune -> point haut jaune
  - distance convertie avec disque 45 cm
  - temps calcule avec FPS source de la video laterale
- Les resultats et le dessin lateral restent affiches une fois la phase terminee.

Points sensibles:
- La qualite depend beaucoup de la detection du disque et de son diametre dans
  l'image.
- Les videos face/laterale ne sont pas synchronisees: la logique est decouplee,
  mais l'interpretation humaine doit garder ce point en tete.
- La detection de pieds peut encore demander un ajustement selon cadrage,
  chaussures, lumiere et qualite de pose.


Etat actuel - Deadlift
----------------------
Ce qui marche:
- Vue face/corps:
  - detection du mouvement
  - detection d'une redescente visible de face
  - detection pieds
  - affichage angles de genoux
- Vue laterale:
  - disque.pt par defaut pour suivre le disque
  - detection debut de remontee disque
  - detection fin de remontee disque
  - trajectoire de remontee
  - gel du dessin a la fin
  - vitesse moyenne bas -> haut avec FPS lateral
- Verdict actuel base seulement sur:
  - redescente
  - pieds

Ce qui est affiche mais pas encore utilise pour refuser:
- Verrouillage des genoux.

Ce qui n'est pas encore traite:
- Lock epaules / verrouillage des epaules en fin de deadlift.
- Appui de la barre sur les cuisses.
- Validation complete du lockout competition avec epaules + hanches + genoux.


Comment est calculee la vitesse disque
--------------------------------------
Pour squat et deadlift, la vitesse est une moyenne de phase:

1. Detecter les points de trajectoire du disque.
2. Chercher le point le plus bas et le point le plus haut de la phase.
3. Mesurer la distance verticale en pixels:

   distance_px = abs(y_bas - y_haut)

4. Convertir en centimetres avec le diametre du disque:

   cm_par_pixel = 45.0 / diametre_disque_px
   distance_cm = distance_px * cm_par_pixel

5. Calculer le temps avec le FPS source de la video laterale:

   temps_s = abs(frame_fin - frame_debut) / fps_lateral

6. Calculer la vitesse:

   vitesse_cm_s = distance_cm / temps_s

Le FPS traitement n'est pas utilise pour ce calcul.


Verrouillage du genou en deadlift
---------------------------------
Le principe recommande:
- Attendre la fin du mouvement ou une position finale candidate stable.
- Prendre une petite fenetre de frames a la fin.
- Calculer l'angle des genoux gauche/droite avec les keypoints de la vue face.
- Utiliser le plus petit angle visible.
- Si l'angle est superieur ou egal au seuil, par exemple 172 degres, alors
  verrouillage OK.
- Sinon, genoux non verrouilles.

Le code a deja une base avec DEADLIFT_LOCKOUT_KNEE_ANGLE = 172.0. Pour l'instant,
le verdict final ne l'utilise pas, car le verdict a ete limite volontairement a
redescente + pieds.


Lock epaules et appui cuisse en deadlift
----------------------------------------
Ces fautes ne sont pas encore implementees.

Piste pour le lock epaules:
- Utiliser les keypoints epaules/hanches sur la vue face ou un modele lateral.
- Detecter la position finale stable.
- Verifier que les epaules sont revenues en arriere/au-dessus de la ligne des
  hanches selon une tolerance definie.
- Ajouter une fenetre de stabilite comme pour les genoux, pour eviter de juger
  sur une seule frame.

Piste pour l'appui cuisse:
- Il faut une vue laterale exploitable.
- Suivre la barre/disque et le segment cuisse via keypoints hanche/genou.
- Detecter un contact prolonge ou une trajectoire qui reste collee a la cuisse.
- Ce point est plus difficile que les autres, car il demande une bonne estimation
  de distance barre-cuisse et une tolerance selon le cadrage.


Developpe couche / bench
------------------------
Le bench n'est pas traite pour l'instant.

L'interface contient "Developpe couche", mais le lancement affiche un message
indiquant que le moteur n'est pas encore disponible. Il faudra definir:
- les vues necessaires;
- les modeles de detection utiles;
- les phases: depart, descente, pause poitrine, remontee, verrouillage;
- les fautes: absence de pause, rebond, fesses levees, pieds, trajectoire,
  lockout incomplet.


Ce qu'il reste a faire
----------------------
Priorites possibles:

1. Stabiliser encore la detection des pieds.
   Ajouter un mode debug visuel qui affiche les points de reference et le score
   de deplacement directement sur la video.

2. Finaliser le verrouillage genou deadlift.
   Decider si le verdict doit inclure cette faute et tester sur plusieurs videos.

3. Ajouter le lock epaules et l'appui cuisse en deadlift.
   Ces controles ne sont pas encore implementes.

4. Ajouter le developpe couche.
   Il est present dans l'interface, mais pas implemente.

5. Ajouter une exportation des resultats.
   Par exemple fichier CSV ou JSON avec frames, vitesses, verdicts et fautes.

6. Mieux gerer 3 vues.
   Aujourd'hui la logique principale utilise surtout les deux premieres vues.

7. Ajouter des tests automatises.
   Les analyseurs sont tres lies aux videos et modeles YOLO. Des tests unitaires
   peuvent quand meme couvrir les calculs de vitesse, phases laterales et
   fonctions geometriques.

8. Nettoyer les textes encodes.
   Certains environnements affichent mal les accents selon l'encodage console.


Conseils pour continuer le projet
---------------------------------
- Ne pas recoupler la vue face et la vue laterale: chaque vue doit produire ses
  propres reperes temporels quand elle mesure une trajectoire.
- Toujours utiliser le FPS source de la vue qui fournit la mesure.
- Pour les vitesses physiques, preferer les extremums de trajectoire de phase
  plutot que les premiers/derniers points detectes, car les detections peuvent
  commencer ou finir avec du bruit.
- Garder les seuils dans constants.py pour faciliter les reglages.
- Verifier chaque changement avec au moins une video squat et une video deadlift.
