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
from PIL import Image


class Detector:

    def __init__(self):
        self.cfg = get_cfg()  # Configuracion por defecto
        self.cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.WEIGHTS = "model_final.pth"  # Carga del fichero modelo
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confianza limite, posterior edicion
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    def inference(self, fichero):
        print(img)

        im = cv2.imread(fichero)

        predictor = DefaultPredictor(self.cfg)
        # imagen = cv2.imread(fichero)
        outputs = predictor(im)
        salida = outputs['instances']

        # metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        v = Visualizer(im[:, :, ::-1])
        v = v.draw_instance_predictions(salida.to("cpu"))
        resultado = v.get_image()[:, :, ::-1]

        return resultado

class Detector:

    def __init__(self):

        self.cfg = get_cfg() # Configuracion por defecto
        self.cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.WEIGHTS = "model_final.pth" # Carga del fichero modelo
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # Confianza limite, posterior edicion
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    def inference(self, fichero):
        im = cv2.imread(fichero)

        predictor = DefaultPredictor(self.cfg)
        #imagen = cv2.imread(fichero)
        outputs = predictor(im)
        salida = outputs['instances']

        # metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        v = Visualizer(im[:, :, ::-1])
        v = v.draw_instance_predictions(salida.to("cpu"))
        resultado = v.get_image()[:, :, ::-1]

        return resultado