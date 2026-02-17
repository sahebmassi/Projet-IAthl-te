from tkinterObjects import *
from multiThreading import *
from tkinterOutputData import *
import os
from avoirVideo import *

from onnxOptimiser import optimiser
from tkinter import messagebox


BASE_MODEL_POSE = "yolov8n-pose.pt"  # Changé de yolov8n-pose à yolov8m-pose
BASE_MODEL_CUSTOM = "best.pt"


def creationFenetreChoix(nbArbitre : int, choixTrouverMateriel : str, models_opti : list) -> None:
    """Créer la fenêtre pour choisir les paramètres de chaque arbitre
    Args:
        nbArbitre (int): le nombre d'arbitre
        choixTrouverMateriel (str): le matériel à utiliser"""
    fenetreChoix = tk.Tk()
    fenetreChoix.title("Configuration des arbitres")
    fenetreChoix.geometry("1000x900")
    scrollable_frame = creationScrollableFrame(fenetreChoix)
    choices = []
    for i in range(nbArbitre):
        frame = tk.Frame(scrollable_frame, padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)
        choixTraitementVideo = choixMenuDeroulant(frame, f"Choix du traitement vidéo (arbitre {i+1}):", 0, 0, multipleChoixTraitementVideo, informationChoixTraitementVideo)
        
        tk.Label(frame, text=f"Enregistrer les données dans un fichier ? (arbitre {i+1}) :").grid(row=1, column=0, sticky="w")
        boolEnregistrementJson = tk.StringVar(value="Non")
        boutonOuiEnregistrementJson = tk.Radiobutton(frame, text="Oui", variable=boolEnregistrementJson, value="Oui")
        boutonNonEnregistrementJson = tk.Radiobutton(frame, text="Non", variable=boolEnregistrementJson, value="Non")
        boutonOuiEnregistrementJson.grid(row=1, column=1, sticky="w")
        boutonNonEnregistrementJson.grid(row=1, column=2, sticky="w")
        enregistrementJson = tk.Text(frame, height=1, state="disabled")
        enregistrementJson.grid(row=2, column=1, columnspan=2, pady=5)
        
        choixTrouverArticulation = choixMenuDeroulant(frame, f"Choix pour trouver les articulations (arbitre {i+1}):", 3, 0, multipleChoixTrouverArticulation, informationChoixTrouverArticulation)        
        
        tk.Label(frame, text=f"Path de la vidéo (arbitre {i+1}):").grid(row=4, column=0, sticky="w")
        videoPath = ttk.Combobox(frame, values=multipleChoixCamera,state="normal")
        videoPath.grid(row=4, column=1)
        tabChoixCamera.append(videoPath)
        
        parcourirFichier = tk.Button(frame, text="Parcourir", command=lambda e=videoPath: choisirFichierDossier(e))
        parcourirFichier.grid(row=4, column=2)
        info = tk.Button(frame, text="?", command=lambda: afficheInformation(None, informationPathVideo))
        info.grid(row=4, column=3)
        
        tk.Label(frame, text=f"Enregistrer la vidéo avant le traitement ? (arbitre {i+1}):").grid(row=5, column=0, sticky="w")
        boolEnregistrementVideoSansTraitement = tk.StringVar(value="Non")
        boutonOuiEnregistrementVideoSansTraitement = tk.Radiobutton(frame, text="Oui", variable=boolEnregistrementVideoSansTraitement, value="Oui")
        boutonNonEnregistrementVideoSansTraitement = tk.Radiobutton(frame, text="Non", variable=boolEnregistrementVideoSansTraitement, value="Non")
        boutonOuiEnregistrementVideoSansTraitement.grid(row=5, column=1, sticky="w")
        boutonNonEnregistrementVideoSansTraitement.grid(row=5, column=2, sticky="w")
        
        enregistrementVideoSansTraitement = tk.Text(frame, height=1, state="disabled")
        enregistrementVideoSansTraitement.grid(row=6, column=1, columnspan=2, pady=5) 

        tk.Label(frame, text=f"Enregistrer la vidéo après le traitement ? (arbitre {i+1}):").grid(row=7, column=0, sticky="w")
        boolEnregistrementVideoAvecTraitement = tk.StringVar(value="Non")
        boutonOuiEnregistrementVideoAvecTraitement = tk.Radiobutton(frame, text="Oui", variable=boolEnregistrementVideoAvecTraitement, value="Oui")
        boutonNonEnregistrementVideoAvecTraitement = tk.Radiobutton(frame, text="Non", variable=boolEnregistrementVideoAvecTraitement, value="Non")
        boutonOuiEnregistrementVideoAvecTraitement.grid(row=7, column=1, sticky="w")
        boutonNonEnregistrementVideoAvecTraitement.grid(row=7, column=2, sticky="w")
        
        enregistrementVideoAvecTraitement = tk.Text(frame, height=1, state="disabled")
        enregistrementVideoAvecTraitement.grid(row=8, column=1, columnspan=2, pady=5) 
        
        def choixEnregistrementVideoAvecTraitement():
            """Vérifie l'état des boutons pour savoir si on peut écrire un path
            """
            for i in range(len(choices)):
                if choices[i][6].get() == "Oui":
                    choices[i][7].config(state="normal")
                else:
                    choices[i][7].config(state="disabled")
                    choices[i][7].delete("1.0", "end")
        
        def choixEnregistrementVideoSansTraitement():
            """Vérifie l'état des boutons pour savoir si on peut écrire un path
            """
            for i in range(len(choices)):
                if choices[i][3].get() == "Oui":
                    choices[i][4].config(state="normal")
                else:
                    choices[i][4].config(state="disabled")
                    choices[i][4].delete("1.0", "end")

        def choixEnregistrementJson():
            """Vérifie l'état des boutons pour savoir si on peut écrire un path
            """
            for i in range(len(choices)):
                if choices[i][8].get() == "Oui":
                    choices[i][5].config(state="normal")
                else:
                    choices[i][5].config(state="disabled")
                    choices[i][5].delete("1.0", "end")
        
        boutonOuiEnregistrementVideoAvecTraitement.config(command=choixEnregistrementVideoAvecTraitement)
        boutonNonEnregistrementVideoAvecTraitement.config(command=choixEnregistrementVideoAvecTraitement)
        boutonOuiEnregistrementVideoSansTraitement.config(command=choixEnregistrementVideoSansTraitement)
        boutonNonEnregistrementVideoSansTraitement.config(command=choixEnregistrementVideoSansTraitement) 
        boutonOuiEnregistrementJson.config(command=choixEnregistrementJson)
        boutonNonEnregistrementJson.config(command=choixEnregistrementJson)
        choices.append((choixTraitementVideo, choixTrouverArticulation, videoPath, boolEnregistrementVideoSansTraitement, 
                        enregistrementVideoSansTraitement, enregistrementJson, boolEnregistrementVideoAvecTraitement, 
                        enregistrementVideoAvecTraitement, boolEnregistrementJson))
    
    def validate_choices():
        """action faites quand le bouton valider est appuyer
        """
        results = []
        for i, (choixTraitementVideo, choixTrouverArticulation, videoPath, boolEnregistrementVideoSansTraitement, 
                enregistrementVideoSansTraitement, enregistrementJson, boolEnregistrementVideoAvecTraitement, 
                enregistrementVideoAvecTraitement, boolEnregistrementJson) in enumerate(choices):
            
            textEnregistrementVideoSansTraitement = enregistrementVideoSansTraitement.get("1.0", "end").strip() if boolEnregistrementVideoSansTraitement.get() == "Oui" else ""
            textEnregistrementVideoAvecTraitement = enregistrementVideoAvecTraitement.get("1.0", "end").strip() if boolEnregistrementVideoAvecTraitement.get() == "Oui" else ""
            textEnregistrementJson = enregistrementJson.get("1.0", "end").strip() if boolEnregistrementJson.get() == "Oui" else ""
            
            materiel = choixMaterielOutput(choixTrouverMateriel)
            traitementVideo = choixTraitementVideoOutput(str(choixTraitementVideo.get()))
            trouverArticulation = choixTrouverArticulationOutput(str(choixTrouverArticulation.get()))
            enregistrementVideoAvant = None if boolEnregistrementVideoSansTraitement.get()=="Non" else textEnregistrementVideoSansTraitement
            enregistrementVideoApres = None if boolEnregistrementVideoAvecTraitement.get()=="Non" else textEnregistrementVideoAvecTraitement
            enregistrementJson = None if boolEnregistrementJson.get() == "Non" else textEnregistrementJson
            pathVideoOutput = str(videoPath.get())
            if os.path.isdir(pathVideoOutput):
                if verifierInformation(traitementVideo, trouverArticulation, enregistrementVideoAvant, enregistrementVideoApres, enregistrementJson):
                    nbVid = 0
                    for fichier in os.listdir(pathVideoOutput):
                        if estVideo(fichier):
                            enregistrementVideoApresTemp = str(nbVid) if enregistrementVideoApres != None else None
                            enregistrementVideoAvantTemp = str(nbVid) if enregistrementVideoAvant != None else None
                            nbVid += 1
                            captureVideo(mediaPipe=False, yolo = models_opti, trouverArticulation=trouverArticulation, videoPath=pathVideoOutput + "/" + fichier, enregistrerVideoAvant=enregistrementVideoAvantTemp, traitementVideo=traitementVideo, choixMateriel=materiel, enregistrementVideoApres=enregistrementVideoApresTemp, enregistrementJson=enregistrementJson)
                            # results.append([
                            #     False,
                            #     models_opti,
                            #     trouverArticulation,
                            #     pathVideoOutput + "/" + fichier,
                            #     enregistrementVideoAvantTemp,
                            #     traitementVideo,
                            #     materiel,
                            #     enregistrementVideoApresTemp,
                            #     enregistrementJson
                            # ])
                else :
                    return

            elif estVideo(pathVideoOutput) or pathVideoOutput.isdigit():
                if verifierInformation(traitementVideo, trouverArticulation, enregistrementVideoAvant, enregistrementVideoApres, enregistrementJson):
                    captureVideo(mediaPipe=False, yolo = models_opti, trouverArticulation=trouverArticulation, videoPath=pathVideoOutput, enregistrerVideoAvant=enregistrementVideoAvant, traitementVideo=traitementVideo, choixMateriel=materiel, enregistrementVideoApres=enregistrementVideoApres, enregistrementJson=enregistrementJson)
                    # results.append([
                    #     False,
                    #     models_opti,
                    #     trouverArticulation,
                    #     pathVideoOutput,
                    #     enregistrementVideoAvant,
                    #     traitementVideo,
                    #     materiel,
                    #     enregistrementVideoApres,
                    #     enregistrementJson
                    # ])
                else :
                    return

            else :
                messagebox.showerror("Erreur", "Veuillez entrer un fichier video ou un dossier dans videopath.")
                return
        cameraCreator(results, models_opti)
    tk.Button(scrollable_frame, text="Valider", command=validate_choices).pack(pady=10)
    fenetreChoix.mainloop()

def main() -> None:
    """Lance la fenêtre de base pour choisir les arbitres
    """
    root = tk.Tk()
    root.title("Nombre d'arbitre")
    root.geometry("600x100")
    
    tk.Label(root, text="Combien d'arbitre voulez-vous lancer ?").grid(row=0, column=0)
    thread_var = tk.StringVar()
    
    entree = tk.Entry(root, textvariable=thread_var)
    entree.grid(row=0, column=1)
    
    choixTrouverMateriel = choixMenuDeroulant(root, f"Choix pour le matériel utilisé pour le lancement", 1, 0, multipleChoixMateriel, informationChoixMateriel)        

    def validate():

        try:

            if str(thread_var.get()).isdigit():
                nbArbitre = int(thread_var.get())


                if nbArbitre > 0:
                    choixMateriel = choixTrouverMateriel.get()

                    if not choixMateriel:
                        messagebox.showerror("Erreur", "Veuillez entrer un matériel valide.")
                        return
                    
                    # --- DBT DE L'OPTIMISATION ---
                    try:

                        optimised_model_pose = optimiser(BASE_MODEL_POSE)
                        optimised_model_custom = optimiser(BASE_MODEL_CUSTOM)
                        optimised_models = [optimised_model_pose, optimised_model_custom]
                        print("Optimisation terminée.")
                    except Exception as e:
                        messagebox.showerror("Erreur d'optimisation", f"Impossible d'optimiser les modèles : {e}")
                        return
                    # --- FIN DE L'OPTIMISATION ---
                    root.destroy()
                    creationFenetreChoix(nbArbitre, choixMateriel, optimised_models)

                else:
                    messagebox.showerror("Erreur", "Veuillez entrer un nombre valide d'arbitre.")
            else:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre valide d'arbitre.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre entier.")
    
    tk.Button(root, text="Valider", command=validate).grid(row=2, column=1)

    root.mainloop()

if __name__ == "__main__":
    main()

    #nettoyage des fichiers temp aws
    for tmp_path in temp_files_to_clean:
        if tmp_path and os.path.exists(tmp_path):
            try :
                os.remove(tmp_path)
            except OSError as e :
                print("erreur nettoyage")
