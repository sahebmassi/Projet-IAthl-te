import multiprocessing
from avoirVideo import captureVideo
from traitementVideo import *
from trouverArticulationYolo import *
# from onnxOptimiser import *

threadingList = []

def cameraCreator(myListe : list, modelsOpti : list) -> None:
    """Lance les process de caméra
    Args:
        myListe (list): liste des argument pour lancer capture vidéo
        models (list): liste des modèles
    """
    # modelsOpti = [optimiser(models[0]), optimiser(models[1])]
    processList = [multiprocessing.Process(target = captureVideo, 
                                       kwargs={
                                                "mediaPipe": elem[0],
                                                "yolo" : modelsOpti,
                                                "trouverArticulation" : elem[2],
                                                "videoPath" : elem[3],
                                                "enregistrerVideoAvant" : elem[4],
                                                "traitementVideo" : elem[5],
                                                "choixMateriel" : elem[6],
                                                "enregistrementVideoApres" : elem[7],
                                                "enregistrementJson" : elem[8],
                                                "id" : myListe.index(elem)
                                            }) for elem in myListe]
    for p in processList:
        p.start()
    try:
        for p in processList:
            p.join()
    except KeyboardInterrupt:
        print("key received")
        # for p in processList:
        #      p.join()
        pass
    finally : #nettoyage
        for p in processList :
            if p.is_alive():
                # p.terminate() 
                p.kill() 
                p.join()