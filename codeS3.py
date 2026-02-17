import boto3
from botocore.client import Config
import cv2
import requests
import tempfile

def downloadTempVideoS3(bucket : any, key : any) -> str:
    """Télécharge une vidéo temporaire à partir d'AWS
    Args:
        bucket (any): bucket source
        key (any): clé d'autorisation
    Returns:
        str: path vers la vidéo temporaire"""
    # Configuration du client S3
    s3 = boto3.client('s3', 
                      region_name='eu-west-3',
                      config=Config(signature_version='s3v4'))

    # Générer une URL présignée (valable pendant 1 heure)
    presigned_url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=3600
    )
    print(presigned_url)

     # Télécharger la vidéo temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        local_path = tmp_file.name
        response = requests.get(presigned_url, stream=True)
        
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                tmp_file.write(chunk)
        else:
            print(f"Erreur lors du téléchargement : {response.status_code}")
            return -1

    print(presigned_url)
    print (response)
    
    return local_path


def processVideo(local_path : str) -> None:
    """Visualisation vidéo
    Args:
        local_path (str): le path local de la vidéo"""
    cap = cv2.VideoCapture(local_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


bucket_name = 'freebuckettest5go'
key = 'video/00010.MTS'

downloadTempVideoS3(bucket_name, key)