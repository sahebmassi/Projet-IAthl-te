import torch
import platform
import os

import cv2

# from cv2_enumerate_cameras import enumerate_cameras
# from cv2.videoio_registry import getBackendName
# from cv2_enumerate_cameras import supported_backends

def getDevices() -> list :
   """permet de récupérer la liste des GPUs disponible et d'ajouter l'option CPU à la liste du matériel disponible
   Returns:
       list: liste des gpu et du cpu"""
   listMat = []
   for i in range(torch.cuda.device_count()):
      listMat.append(torch.cuda.get_device_properties(i).name)
   listMat.append("Plateforme "+str(platform.processor())+" avec "+str(os.cpu_count())+" coeurs")
   return listMat

def getCameras() -> list :
   """permet de récupérer la liste des cameras disponible
   Returns:
       list: liste des caméras"""
   listCam = []

   for i in range (10) :
      cap = None

      try :

         cap = cv2.VideoCapture(i)

         if cap and cap.isOpened():
               print(f"  Caméra à l'index {i}")
               listCam.append(i)
      
      except Exception as e :
         print( f"une exception {e} est ressortie à l'index {i}")

      finally :
         if cap and cap.isOpened():
            cap.release()

   return listCam
