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

    def __init__(self):
        self.cfg = get_cfg()  # Configuracion por defecto
        self.cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.WEIGHTS = "model_final.pth"  # Carga del fichero modelo
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # Confianza limite, posterior edicion
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    def inference(self, fichero):
        print(fichero)
        im = cv2.imread(fichero)
        predictor = DefaultPredictor(self.cfg)
        outputs = predictor(im)

        salida = outputs['instances']

        v = Visualizer(im[:, :, ::-1])
        v = v.draw_instance_predictions(salida.to("cpu"))
        resultado = v.get_image()[:, :, ::-1]

        nombreDeteccion = fichero[:-4] + '_DETECTED.png'
        print(nombreDeteccion)
        cv2.imwrite(nombreDeteccion, resultado)

        return salida

    def getOutputMask(self, salida):
        pred = salida.get('pred_masks')
        pred = pred.to("cpu").numpy()
        mask = pred[0].astype(bool)
        maskAux = np.zeros_like(mask)

        for i in range(len(pred)):
            mask = pred[i].astype(bool)
            maskAux = maskAux + pred[i]
            maskAux[maskAux > 1] = 1

        mask = maskAux.astype(bool)

        return mask

    def getConfidence(self, salida):
        porcentajes = salida.get('scores')

        return porcentajes.tolist()

    def getClassList(self, salida):
        # Etiquetado por tamaño de cada defecto

        pred = salida.get('pred_masks')
        pred = pred.to("cpu").numpy()
        mask = pred[0].astype(bool)
        maskAux = np.zeros_like(mask)

        for i in range(len(pred)):
            mask = pred[i].astype(bool)
            maskAux = maskAux + pred[i]
            maskAux[maskAux > 1] = 1

        mask = maskAux.astype(bool)

        sizes = []

        for i in range(len(pred)):  # Si hay mas de un defecto, recorrer instancias y unir imagenes
            mask = pred[i].astype(bool)
            labelImagen = label(mask)

            for prop in regionprops(labelImagen):
                if prop.area >= 200:
                    sizes.append('Big')
                elif prop.area >= 100:
                    sizes.append('Medium')
                else:
                    sizes.append('Small')

        return sizes


