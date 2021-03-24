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
import csv
import cv2
from detector import Detector
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, send_from_directory, send_file, url_for, session
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology
from skimage.measure import label, regionprops
import numpy as np
import datetime
from collections import Counter
import pandas as pd

# Instanciando la clase Flask con el nombre de la app

app = Flask(__name__, static_folder='uploads')
app.secret_key = "es un secreto"
root = os.path.abspath(os.getcwd())

app.config['UPLOAD_FOLDER'] = './uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# El decorador route indica que se reescribe el comportamiento de una funcion que ya existe
@app.route('/')
def mainpage():
    return render_template('index.html', buttonsRes=False)


@app.route("/detect", methods=['POST'])
def upload_image():
  fecha = '{:%H:%M:%S:%d-%m-%Y}'.format(datetime.datetime.now())
  fechaRegistro = '{:%d-%m-%Y}'.format(datetime.datetime.now())
  # Subid imagen y comprobacion de fichero
  if 'image' not in request.files:
	  return render_template("index.html")
  else:
    if request.files:

      # Proceado de imagen y guarado de la imagen original

      f = request.files['image']
      filename = f.filename
      if not filename:
        return render_template('index.html')
      target = os.path.join(root, "uploads/")
      destino = target+filename
      f.save(destino)
      dt = Detector()
      salida = dt.inference(destino)
      im = cv2.imread(destino)
      alturaImagen = im.shape[0]
      anchuraImagen = im.shape[1]

      # Extraccion de mascaras
      mask = dt.getOutputMask(salida)

      # Extraccion de porcentajes
      confs = dt.getConfidence(salida)

      # Extraccion de tipos
      types = dt.getClassList(salida)

      # Procesado de las mascaras binarias generadas y generacion del grafico Plotly

      labels = measure.label(mask)
      fig = px.imshow(im)
      fig.update_traces(hoverinfo='skip')  # hover is only for label info
      props = measure.regionprops(labels, mask)
      properties = ['area']

      for index in range(0, len(confs)):
        label = props[index].label
        contour = measure.find_contours(labels == label, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        hoverinfo += f'<b>{"Área"}: {getattr(props[index], "area"):.2f}</b><br>'
        hoverinfo += f'<b>{"Confianza"}: {confs[index]:.2%}</b><br>'
        hoverinfo += f'<b>{"Tipo"}: {types[index]}</b><br>'
        fig.add_trace(go.Scatter(
          x=x, y=y, name=label,
          mode='lines', fill='toself', showlegend=True,
          hovertemplate=hoverinfo, hoveron='points+fills'))

      nombreGraph = "./uploads/graph_"+f.filename[:-4]+"_"+fecha+".html"
      fig.write_html(nombreGraph)

      # Generando registro en CSV
      numErrores = Counter(types)

      if os.path.isfile('registro.csv'):
        dfc = pd.read_csv('registro.csv')
        df = dfc.copy()

        if fechaRegistro in df.Fecha.unique():
          for i in range(len(df.Fecha)):
            if df.Fecha[i] == fechaRegistro:
              df.iat[i,2] = df.Small[i] + numErrores['Small']
              df.iat[i,3]= df.Medium[i] + numErrores['Medium']
              df.iat[i,4] = df.Big[i] + numErrores['Big']

          df.to_csv("registro.csv", index=False)

        else:
          registro = {'Fecha':fechaRegistro, 'Small':numErrores['Small'], 'Medium':numErrores['Medium'], 'Big':numErrores['Big']}
          registro = df.append(registro, ignore_index=True)
          registro.to_csv('registro.csv')
      else:
        registro = pd.DataFrame([[fechaRegistro, numErrores['Small'], numErrores['Medium'], numErrores['Big']]],
                                columns=['Fecha', 'Small', 'Medium', 'Big'])
        registro.to_csv('registro.csv')

      # Generacion del grafico de lineas Plotly
      dfc = pd.read_csv('registro.csv')
      df = dfc.copy()

      fig = px.line(df, x="Fecha", y=["Small","Medium","Big"],
                    title="Historico de Detecciones",
                    labels={
                     "variable": "Tipos de defecto",
                     "value": "Detecciones"
                 })

      nombreLineas = "./uploads/graphL_" + f.filename[:-4] + "_" + fecha + ".html"
      fig.write_html(nombreLineas)

      session['urlGraphL'] = nombreLineas
      session['widthHTML'] = anchuraImagen
      session['heightHTML'] = alturaImagen

      return render_template('index.html',
                             urlGraph=nombreGraph,
                             deteccion="./uploads/"+f.filename[:-4] + '_DETECTED.png',
                             widthHTML=anchuraImagen*1.25,
                             heightHTML=alturaImagen*1.25,
                             show_hidden=True,
                             buttonsRes=True,
                             histBool=False,
                             urlGraphL= nombreLineas)


    else:
      return render_template("index.html")

@app.route("/hist", methods=['POST'])
def mostrarHist():
  urlHistorico = session['urlGraphL']
  anchuraImagen = session['widthHTML']
  alturaImagen = session['heightHTML']

  return render_template("historico.html",
                         urlGraphL= urlHistorico,
                         widthHTML= anchuraImagen*1.25,
                         heightHTML= alturaImagen*1.25,)


if __name__ == "__main__":
	# run app
	app.run(host='0.0.0.0')