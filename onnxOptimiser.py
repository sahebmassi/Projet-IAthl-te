from ultralytics import YOLO
import onnxruntime as ort


def optimiser (sourcemodel : str) -> any:
    """Optimise le modèle en fonction de l'environement avec ultralytics
    Args:
        sourcemodel (str): le nom du modèle
    Returns:
        any: le modèle optimisé"""
    model = YOLO(sourcemodel)
    modelOpti, providerAvailable = environmentOptimisation(model, sourcemodel)
    return modelOpti

def environmentOptimisation(model : any, sourcemodel : str, useGPU : bool = True)-> list:
    """Adapte l'optimisation par rapport au choix du GPU et à l'environement
    Args:
        model (any): modèle YOLO
        sourcemodel (str): le nom du modèle
        useGPU (bool, optional): utilisation du GPU. Defaults to True.
    Returns:
        list: _description_"""
    liste = sourcemodel.split(".")
    print (sourcemodel)
    print (liste[0])
    name, ext = liste[0], liste[1]
    providers = ort.get_available_providers()
    if useGPU and 'CUDAExecutionProvider' in providers:
        model.export(format="engine", dynamic=True)
        sourceOpti = name+".engine"
        engine_model = YOLO(sourceOpti)        
        return engine_model, ['CUDAExecutionProvider']
    elif 'CoreMLExecutionProvider' in providers:  
        model.export(format="coreml") 
        sourceOpti = name+".mlpackage"
        coreml_model = YOLO(sourceOpti)        
        return coreml_model, ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    else:
        # model.export(format="onnx", dynamic=True, opset = 21)
        # sourceOpti = name + ".onnx"
        # onnx_model = YOLO(sourceOpti)
        onnx_model = YOLO(sourcemodel)
        return onnx_model, ['CPUExecutionProvider']