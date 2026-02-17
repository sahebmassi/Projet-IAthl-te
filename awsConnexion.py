import tkinter as tk
from tkinter import messagebox
from tkinterConfig import *

from awsDownload import *

aws_access_key_id = None
aws_secret_access_key = None
aws_bucket_name = None
aws_key_vids = None

def open_aws_login_window():
    """Ouvre une fenêtre Toplevel pour saisir les identifiants AWS."""
    login_window = tk.Toplevel()
    login_window.title("Identifiants AWS")
    login_window.geometry("500x250") 
    login_window.resizable(False, False)

    lcl_path = None

    main_frame = tk.Frame(login_window, padx=10, pady=10)
    main_frame.pack(expand=True, fill="both")

    tk.Label(main_frame, text="Bucket Name:").grid(row=0, column=0, sticky="w", pady=5)
    bucket_entry = tk.Entry(main_frame, width=35)
    bucket_entry.grid(row=0, column=1, sticky="ew")

    tk.Label(main_frame, text="Video Name:").grid(row=1, column=0, sticky="w", pady=5)
    video_entry = tk.Entry(main_frame, width=35)
    video_entry.grid(row=1, column=1, sticky="ew")

    tk.Label(main_frame, text="Access Key ID:").grid(row=2, column=0, sticky="w", pady=5)
    user_entry = tk.Entry(main_frame, width=35)
    user_entry.grid(row=2, column=1, sticky="ew")

    tk.Label(main_frame, text="Secret Access Key:").grid(row=3, column=0, sticky="w", pady=5)
    pass_entry = tk.Entry(main_frame, show="*", width=35)
    pass_entry.grid(row=3, column=1, sticky="ew")

    if aws_access_key_id:
        user_entry.insert(0, aws_access_key_id)
    if aws_secret_access_key:
        pass_entry.insert(0, aws_secret_access_key)
    if aws_bucket_name:
        bucket_entry.insert(0, aws_bucket_name)
    if aws_key_vids:
        video_entry.insert(0, aws_key_vids)

    def handle_login():
        nonlocal lcl_path

        global aws_access_key_id, aws_secret_access_key, aws_bucket_name, aws_key_vids
        access_key = user_entry.get().strip()
        secret_key = pass_entry.get().strip()
        bucket_key = bucket_entry.get().strip()
        video_key = video_entry.get().strip()

        if not access_key or not secret_key:
             messagebox.showwarning("Champs requis", "Veuillez saisir l'Access Key ID et la Secret Access Key.", parent=login_window)
             return
        
        print("Identifiants AWS stockés.")
        aws_access_key_id = access_key
        aws_secret_access_key = secret_key
        aws_bucket_name = bucket_key
        aws_key_vids = video_key

        handled_lcl_path = downloadTempVideoS3(bucket_key, video_key, access_key, secret_key)

        if handled_lcl_path :
            lcl_path = handled_lcl_path
            login_window.destroy()
            return lcl_path
        else:
            pass


    button_frame = tk.Frame(main_frame)
    button_frame.grid(row=4, column=0, columnspan=2, pady=10)

    validate_button = tk.Button(button_frame, text="Valider", command=handle_login)
    validate_button.pack(side="left", padx=5)

    cancel_button = tk.Button(button_frame, text="Annuler", command=login_window.destroy)
    cancel_button.pack(side="right", padx=5)

    login_window.grab_set()
    login_window.wait_window() 

    return lcl_path 