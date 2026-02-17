import boto3    # <-- AJOUT
from botocore.client import Config # <-- AJOUT
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError # <-- AJOUT pour erreurs boto3
from urllib.parse import urlparse # <-- AJOUT pour parser les URL s3://
import tempfile
from tkinter import messagebox
import os
import requests

# liste des fichiers à nettoyer
from tkinterConfig import temp_files_to_clean

def downloadTempVideoS3(bucket: str, key: str, access_key: str, secret_key: str) -> str:
    """Télécharge une vidéo temporaire depuis AWS S3 dans un fichier local.
    Args:
        bucket (str): Nom du bucket S3.
        key (str): Clé de l'objet (chemin du fichier dans le bucket).
        access_key (str): AWS Access Key ID.
        secret_key (str): AWS Secret Access Key.
    Returns:
        str | None: Chemin vers le fichier vidéo temporaire local, ou None en cas d'erreur.
    """
    print(f"Tentative de téléchargement depuis S3: Bucket='{bucket}', Key='{key}'")
    try:
        # Configuration du client S3 avec les identifiants fournis
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='eu-west-3',
            config=Config(signature_version='s3v4')
        )

        # Générer une URL présignée (valable pendant 1 heure)
        presigned_url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=3600
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file: # Utiliser un suffixe approprié si possible
            
            local_path = tmp_file.name
            print(f"Téléchargement vers le fichier temporaire : {local_path}")

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    local_path = tmp_file.name
                    response = requests.get(presigned_url, stream=True)
                    
                    if response.status_code == 200:
                        for chunk in response.iter_content(chunk_size=1024):
                            tmp_file.write(chunk)
                    else:
                        print(f"Erreur lors du téléchargement : {response.status_code}")
                        return None
                
                temp_files_to_clean.append(local_path)
                return local_path
            
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    messagebox.showerror("Erreur S3", f"La clé '{key}' n'existe pas dans le bucket '{bucket}'.")
                elif e.response['Error']['Code'] == 'NoSuchBucket':
                     messagebox.showerror("Erreur S3", f"Le bucket '{bucket}' n'existe pas ou vous n'y avez pas accès.")
                elif 'Access Denied' in str(e):
                     messagebox.showerror("Erreur S3", f"Accès refusé pour télécharger {key} depuis {bucket}. Vérifiez les permissions IAM.")
                else:
                    messagebox.showerror("Erreur S3", f"Erreur Boto3 lors du téléchargement: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                return None
            
            except Exception as e:
                 messagebox.showerror("Erreur Téléchargement", f"Erreur inattendue lors du téléchargement S3: {e}")
                 if os.path.exists(local_path):
                     os.remove(local_path)
                 return None
            
            # finally :
            #     pass

    except (NoCredentialsError, PartialCredentialsError):
        messagebox.showerror("Erreur AWS", "Identifiants AWS non trouvés ou incomplets.")
        return None
    
    except Exception as e:
        messagebox.showerror("Erreur Client S3", f"Impossible de créer le client S3: {e}")
        return None
