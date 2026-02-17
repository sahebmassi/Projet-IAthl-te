import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinterConfig import *
from awsConnexion import *


def afficheInformation(event: tk.Event, text: str) -> None:
    """Affiche une boîte de dialogue contenant une information.
    Args:
        event (tk.Event): L'événement déclencheur.
        text (str): Le texte d'information à afficher."""
    messagebox.showinfo("Information", text)

def ajouterPath(path : str, entree : tk.Entry) -> None :
    if path:
        if path not in multipleChoixCamera:
            multipleChoixCamera.append(path)
            for elem in tabChoixCamera :
                elem["values"] = multipleChoixCamera
        entree.set(path)

def choisirFichierDossier(entree: tk.Entry) -> None:
    """Ouvre une boîte de dialogue pour sélectionner un fichier et met à jour l'entrée avec son chemin relatif.
    Args:
        entree (tk.Entry): Champ d'entrée où afficher le chemin du fichier sélectionné."""
    root = tk.Tk()
    root.title("Sélecteur de chemin")
    root.geometry("400x200")
    def choisir_fichier():
        global choixPathVideoTemp
        fichier = filedialog.askopenfilename()
        if fichier:
            ajouterPath(fichier, entree)
            root.destroy()
    def choisir_dossier():
        global choixPathVideoTemp
        dossier = filedialog.askdirectory()
        if dossier:
            ajouterPath(dossier, entree)
            root.destroy()
    def choisir_aws():
        global choixPathVideoTemp
        url = open_aws_login_window()
        if url:
            print( "L'url de la video temporaire est : ", url)
            ajouterPath(url, entree)
            root.destroy()

    btn_fichier = tk.Button(root, text="Choisir un fichier", command=choisir_fichier)
    btn_fichier.pack(pady=10)
    btn_dossier = tk.Button(root, text="Choisir un dossier", command=choisir_dossier)
    btn_dossier.pack(pady=10)
    btn_aws = tk.Button(root, text="AWS", command=choisir_aws)
    btn_aws.pack(pady=10)
    root.mainloop()



def choixMenuDeroulant(frame, titre, row, column, valeurChoix, valeurInformation):
    """Crée un menu déroulant avec un bouton d'information.
    Args:
        frame (tk.Frame): Le cadre parent du menu déroulant.
        titre (str): Le titre du menu déroulant.
        row (int): Ligne dans la grille où placer le widget.
        column (int): Colonne dans la grille où placer le widget.
        valeurChoix (list[str]): Liste des options disponibles.
        valeurInformation (str): Texte d'information affiché en cas de besoin.
    Returns:
        ttk.Combobox: Le menu déroulant créé."""
    tk.Label(frame, text=titre).grid(row=row, column=column, sticky="w")
    choice = ttk.Combobox(frame, values=valeurChoix,state="readonly")
    choice.grid(row=row, column=column + 1)
    info = tk.Button(frame, text="?", command=lambda: afficheInformation(None, valeurInformation))
    info.grid(row=row, column=3)
    return choice


def creationScrollableFrame(fenetreChoix):
    """Crée un cadre scrollable dans la fenêtre donnée.
    Args:
        fenetreChoix (tk.Tk): La fenêtre où ajouter le cadre scrollable.
    Returns:
        tk.Frame: Le cadre scrollable."""
    canvas = tk.Canvas(fenetreChoix)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(fenetreChoix, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollable_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    return scrollable_frame