"""
Autor: Fco Javier Yagüe Izquierdo
Año: 2021
Versión: 1.0

Descripción:
Aplicación Web en Flask que junto con Detectron 2 y el modelo entrenado para el mismo, procesa y aplica una detección
sobre una imagen facilitada por el usuario.
"""


import io
import requests
import os
import urllib.request
import os
import cv2
from dt2 import Detector
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, send_from_directory, send_file, url_for
from PIL import Image
# Instanciando la clase Flask con el nombre de la app

app = Flask(__name__, static_folder='uploads')
root = os.path.abspath(os.getcwd())

app.config['UPLOAD_FOLDER'] = './uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# El decorador route indica que se reescribe el comportamiento de una funcion que ya existe
@app.route('/')
def mainpage():
    return render_template('index.html')


@app.route("/detect", methods=['POST'])
def upload_image():
  if 'image' not in request.files:
	  return render_template("index.html")
  else:
    if request.files:
      f = request.files['image']
      filename = f.filename
      if not filename:
        return render_template('index.html')
      target = os.path.join(root, "uploads/")

      destino = target+filename
      f.save(destino)
      dt = Detector()
      resultado = dt.inference(destino)

      nombreDeteccion = filename[:-4] + '_DETECTED.png'
      cv2.imwrite(target+nombreDeteccion, resultado)

      return render_template('index.html', filename=nombreDeteccion, show_hidden=True)

    else:
      return render_template("index.html")

if __name__ == "__main__":
	# run app
	app.run(host='0.0.0.0')