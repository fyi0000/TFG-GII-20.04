"""
Autor: Fco Javier Yagüe Izquierdo
Año: 2021
Versión: 1.0
Descripción:
Aplicación Web en Flask que junto con Detectron 2 y el modelo entrenado para el mismo, procesa y aplica una detección
sobre una imagen facilitada por el usuario.
"""


import io
import os
import csv
import cv2
import requests
import urllib.request
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from detector import Detector
from PIL import Image
from werkzeug.utils import secure_filename
from skimage import data, filters, measure, morphology
from skimage.measure import label, regionprops
from collections import Counter
from io import BytesIO
import base64
from werkzeug.wsgi import FileWrapper
import matplotlib.pyplot as plt
from github import Github
import urllib, json
import gdown
import re
import wget

from flask import Flask, flash, redirect, url_for, render_template, session, request, after_this_request, flash, send_file, jsonify

app = Flask(__name__, static_folder='static')
app.secret_key = 'random secret key'

root = os.path.abspath(os.getcwd())

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# El decorador route indica que se reescribe el comportamiento de una funcion que ya existe
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("deteccion.html", mostrar_preview=False, mostrar_resultado=False, uploading=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template("deteccion.html")
    else:
        if request.files:
          f = request.files['image']
          filename = f.filename

          if not filename:
            return render_template('deteccion.html')

          session['filename'] = filename
          target = os.path.join(root, "static/uploads")
          destino = os.path.join(target, filename)
          preview="./static/uploads/"+filename
          f.save(destino)
          session['destino'] = destino
          session['filename'] = f.filename

          return render_template("deteccion.html", mostrar_preview=True, mostrar_resultado=False, uploading=True, deteccion=preview)

@app.route("/deteccion", methods=['POST'])
def deteccion():
    fecha = '{:%H:%M:%S:%d-%m-%Y}'.format(datetime.datetime.now())
    fechaRegistro = '{:%d-%m-%Y}'.format(datetime.datetime.now())

    confianza = request.form['slider']

    confianza= int(confianza)/100
    destino = session['destino']
    filename = session['filename']
    modelo = session['modelo']

    
    dt = Detector(modelo, confianza)
    salida = dt.inference(destino)
    im = cv2.imread(destino)
    alturaImagen = im.shape[0]
    anchuraImagen = im.shape[1]

    # Extraccion de mascaras
    mask = dt.getOutputMask(salida)
    if mask == 'Sin Defectos':
        fig = px.imshow(im)
        fig.update_traces(hoverinfo='skip')
        nombreGraph = "./static/uploads/graph_" + filename[:-4] + "_" + fecha + ".html"
        fig.write_html(nombreGraph)

        session['resultado'] = nombreGraph
        session['widthHTML'] = anchuraImagen
        session['heightHTML'] = alturaImagen

        return render_template("deteccion.html",
                           mostrar_preview=False,
                           uploading=False,
                           mostrar_resultado=True,
                           conDefectos = False,
                           result=session['resultado'],
                           anchoGrafico=anchuraImagen,
                           altoGrafico=alturaImagen)


    mascaraImagen = Image.fromarray(mask)

    session['mascaraBinaria'] = './static/uploads/'+filename+'MB.png'
    mascaraImagen.save('./static/uploads/'+filename+'MB.png')

    # Extraccion de porcentajes
    confs = dt.getConfidence(salida)

    # Extraccion de tipos
    types = dt.getClassList(salida)

    # Procesado de las mascaras binarias generadas y generacion del grafico Plotly

    labels = measure.label(mask)
    fig = px.imshow(im)
    fig.update_traces(hoverinfo='skip') 
    props = measure.regionprops(labels, mask)
    properties = ['area']

    for index in range(0, len(confs)-1):
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

    nombreGraph = "./static/uploads/graph_" + filename[:-4] + "_" + fecha + ".html"
    fig.write_html(nombreGraph)

    # Generando registro en CSV
    numErrores = Counter(types)
    numDetecciones = 55
    if os.path.isfile('registro.csv'):
        dfc = pd.read_csv('registro.csv')
        df = dfc.copy()

        if fechaRegistro in df.Fecha.unique():
            for i in range(len(df.Fecha)):
                if df.Fecha[i] == fechaRegistro:
                    numDetecciones = df.iat[i, 0]
                    df.iat[i, 2] = df.Small[i] + numErrores['Small']
                    df.iat[i, 3] = df.Medium[i] + numErrores['Medium']
                    df.iat[i, 4] = df.Big[i] + numErrores['Big']
        print(numDetecciones, 'AAAAAAAAAAAAAAAA')
        df.to_csv("registro.csv", index=False)

        '''
        else:
          registro = {'N':1,'Fecha':fechaRegistro, 'Small':numErrores['Small'], 'Medium':numErrores['Medium'], 'Big':numErrores['Big']}
          registro = df.append(registro, ignore_index=True)
          registro.to_csv('registro.csv')

      else:
        registro = pd.DataFrame([[1,fechaRegistro, numErrores['Small'], numErrores['Medium'], numErrores['Big']]],
                                columns=['N','Fecha', 'Small', 'Medium', 'Big'])
        registro.to_csv('registro.csv')

        '''
    # Generacion del grafico de lineas Plotly
    dfc = pd.read_csv('registro.csv')
    df = dfc.copy()

    fig = px.bar(df, x="Fecha", y=["Small", "Medium", "Big"],
                 title="Historico de Detecciones",
                 text='N',
                 labels={
                     "variable": "Tipos de defecto",
                     "value": "Detecciones"}
                 )

    nombreHistorico = "./static/uploads/graphL_" + filename[:-4] + "_" + fecha + ".html"
    fig.write_html(nombreHistorico)

    session['resultado'] = nombreGraph
    session['urlGraphL'] = nombreHistorico
    session['widthHTML'] = anchuraImagen
    session['heightHTML'] = alturaImagen
    print(session['resultado'])

    return render_template("deteccion.html",
                           mostrar_preview=False,
                           uploading=False,
                           mostrar_resultado=True,
                           conDefectos = True,
                           result=session['resultado'],
                           anchoGrafico=anchuraImagen,
                           altoGrafico=alturaImagen)

@app.route("/descargarDinamico")
def descargarDinamico():
    ficheroHTML = session['resultado']
    filename = session['filename']
    return send_file(ficheroHTML, attachment_filename=filename[:-4] + ".html", as_attachment=True)

@app.route("/descargarComposicion")
def descargarComposicion():
    filename = session['filename']
    original = session['destino']
    mascaraBinaria = session['mascaraBinaria']

    im = Image.open(original)
    im2 = Image.open(mascaraBinaria)
    
    fig = plt.figure(figsize=(100, 100))

    a = fig.add_subplot(2, 1, 1)
    plt.axis('off')
    imgplot = plt.imshow(im)
    plt.axis("tight")
    plt.gca().set_aspect('equal')

    a = fig.add_subplot(2, 1, 2)
    plt.axis('off')
    imgplot = plt.imshow(im2)
    plt.axis("tight")
    plt.gca().set_aspect('equal')

    plt.subplots_adjust(wspace=0, hspace=0)

    tempFig = BytesIO()

    plt.savefig(tempFig, pad_inches=0, bbox_inches='tight', format='png')
    tempFig.seek(0)
    return send_file(tempFig, mimetype="image/png", attachment_filename=filename[:-4] + "_CMP.png", as_attachment=True)

@app.route("/descargarMascara")
def descargarMascara():
    mascaraBinaria = session['mascaraBinaria']
    filename = session['filename']
    return send_file(mascaraBinaria, attachment_filename=filename[:-4] + "_MB.png", as_attachment=True)

@app.route("/hist")
def hist():
    return render_template("historico.html", urlGraph=session['urlGraphL'])

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route('/checkModelVersion', methods=['POST'])
def checkModelVersion():
    ficheros = os.listdir()
    ficheros = [m for m in ficheros if 'modelo' in m and '.pth' in m]
    
    versionLocalMasReciente, versionLocalMasRecienteTxt = getMostRecentVersion(
        ficheros)

    g = Github("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")
    repo = g.get_repo("fyi0000/HolaMundoTarea")
    modelosJson = repo.get_contents("modelos.json")
    dictModelos = json.loads(modelosJson.decoded_content)
    versiones = [int(v['version'].replace('.','')) for v in dictModelos['modelos']]
    versionesTxt = [v['version'] for v in dictModelos['modelos']]

    
    lastVersion = max(versiones)
    lastVersionTxt = versionesTxt[versiones.index(lastVersion)]
    url = str(dictModelos['modelos'][versiones.index(lastVersion)]['url'])
    print('La ultima version es la '+lastVersionTxt+' con un enlace: ', url)
    print('Version entera', lastVersion)


    if versionLocalMasReciente >= lastVersion:
        return jsonify(status = 'Updated', recentVersion=lastVersionTxt, oldVersion = versionLocalMasRecienteTxt)
    
    session['urlUpdate'] = url
    session['versionUpdate'] = lastVersionTxt

    return jsonify(status='Outdated', recentVersion=lastVersionTxt, oldVersion = versionLocalMasRecienteTxt)


@app.route('/updateModel', methods=['POST'])
def updateModel():
    lastVersionTxt = session['versionUpdate']
    url = session['urlUpdate']
    nombreFichero = 'modelo-'+lastVersionTxt+'.pth'
    session['modelo'] = nombreFichero
    gdown.download(url, output='./'+nombreFichero, quiet=True)
    ficheros = os.listdir()
    ficheros = [m for m in ficheros if 'modelo' in m and '.pth' in m]
    
    versionLocalMasReciente, versionLocalMasRecienteTxt = getMostRecentVersion(
        ficheros)

    if (lastVersionTxt == versionLocalMasRecienteTxt):
        return jsonify(status='OK', message='Modelo actualizado a la version '+ versionLocalMasRecienteTxt)        

    return jsonify(status='Error', message='No se ha podido actualizar el modelo.')

def getMostRecentVersion(list):
    masReciente = 0
    for e in list:
        version = re.findall("[\\b\d\\b]", e)
        versionNumber = int(''.join(version))
        if versionNumber > masReciente:
            masReciente = versionNumber
            masRecienteTxt = '.'.join(version)

    return masReciente, masRecienteTxt
if __name__ == "__main__":
    app.run(host='0.0.0.0')
