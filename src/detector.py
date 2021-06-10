"""
Autor: Fco Javier Yagüe Izquierdo
Año: 2021
Versión: 1.0

Descripción:
Clase que implementa los metodos y funciones para crear un objeto Predictor de Detectron2 cargando los pesos del
entrenamiento previo sobre el conjunto de test.

Referencias:

https://github.com/spiyer99/detectron2_web_app/blob/master/ObjectDetector.py

"""

import numpy as np
import torch
import json
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from skimage.measure import label, regionprops
from PIL import Image


class Detector:

    def __init__(self, modelo, confianza=0.7):
        self.cfg = get_cfg()  # Configuracion por defecto
        self.cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.WEIGHTS = modelo  # Carga del fichero modelo
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confianza  # Confianza limite, posterior edicion
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Aplicacion del modelo a la imagen pasada por parametro, se genera una imagen resultado con el subfijo correspondiente
    def inference(self, fichero):
        im = cv2.imread(fichero)
        predictor = DefaultPredictor(self.cfg)
        outputs = predictor(im)

        salida = outputs['instances']

        v = Visualizer(im[:, :, ::-1])
        v = v.draw_instance_predictions(salida.to("cpu"))
        resultado = v.get_image()[:, :, ::-1]

        nombreDeteccion = fichero[:-4] + '_DETECTED.png'
        cv2.imwrite(nombreDeteccion, resultado)

        return salida

    #Extraccion las mascaras resultado de la ejecucion 
    def getOutputMask(self, salida):
        pred = salida.get('pred_masks')
        masks = pred.to("cpu").numpy()
        
        if pred.size == 0: # Si no hay detecciones, valor en cadena para el mensaje de JavaScript
            return 'Sin Defectos'

        return masks

    #Extraccion de la lista de confianza de las detecciones
    def getConfidence(self, salida):
        porcentajes = salida.get('scores')
        return porcentajes.tolist()

   

