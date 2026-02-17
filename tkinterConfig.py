from hardwareInfo import *
#Représente les données pour l'interface
informationChoixTraitementVideo = "Information sur le traitement vidéo"
multipleChoixTraitementVideo = ["rien", "boxes", "articulation", "articulation et boxes", "Developpe Coucher", "Squat", "Soulever Terre"]

informationChoixTrouverArticulation = "Information pour trouver les articulations"
multipleChoixTrouverArticulation = ["yolo", "yolo boxe centree", "yolo boxe souris", "yolo proche barre"]

informationPathVideo = "Information sur le path de la vidéo"
choixPathVideoTemp = ""
#Instancie les données pour les matériaux
informationChoixMateriel = "Information sur le materiel qui peut etre utiliser"
multipleChoixMateriel = getDevices()
multipleChoixCamera = getCameras()
tabChoixCamera = []


temp_files_to_clean = []
