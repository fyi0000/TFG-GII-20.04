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
from plotly.subplots import make_subplots
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
import urllib
import json
import gdown
import re
import wget

from flask import Flask, flash, redirect, url_for, render_template, session, request, after_this_request, flash, send_file, jsonify

# Inicializacion de la web y carpeta de contenido estatico
app = Flask(__name__, static_folder='static') 
app.secret_key = 'random secret key'

root = os.path.abspath(os.getcwd())

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 #Se acorta el cache para evitar que no se recargue el contenido de Plotly
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Maximo tamaño de las imagenes
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg']) #Extensiones admitidas


"""
Comprueba que el fichero tiene las extensiones admitidas
Metodo original de Flask

https://flask.palletsprojects.com/en/0.12.x/patterns/fileuploads/
"""
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Carga de la pantalla de inicio
@app.route("/")
def home():
    return render_template("index.html")

# Acceso a la seccion de carga desde el inicio o la navbar
@app.route("/upload")
def upload():
    return render_template("deteccion.html", mostrar_preview=False, mostrar_resultado=False, uploading=True, fichero_incompatible=False)

# Carga de la imagen con el metodo POST, que carga la imagen y la almacena en el servidor
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files: # Si el contenido de la peticion POST esta vacio, recargar
        return render_template("deteccion.html")
    else:
        if request.files:
          f = request.files['image']
          filename = f.filename
          
          # Si hay un fichero erroneo o de extension no valida, recargar notificando el problema
          if not filename or not allowed_file(filename): 
            return render_template("deteccion.html", mostrar_preview=False, mostrar_resultado=False, uploading=True, fichero_incompatible=True)
          
          # Variables de sesion que almacenan la informacion del fichero actual
          session['filename'] = filename
          target = os.path.join(root, "static/uploads")
          destino = os.path.join(target, filename)
          preview = "./static/uploads/"+filename
          f.save(destino)
          session['destino'] = destino
          session['filename'] = f.filename

          return render_template("deteccion.html", mostrar_preview=True, mostrar_resultado=False, uploading=True, fichero_incompatible=False, deteccion=preview)

# Se ejecuta la detección, instanciando el predictor y generando los gráficos
@app.route("/deteccion", methods=['POST'])
def deteccion():
    fecha = '{:%H:%M:%S:%d-%m-%Y}'.format(datetime.datetime.now())
    fechaRegistro = '{:%d-%m-%Y}'.format(datetime.datetime.now())

    confianza = request.form['slider'] # Valor del slider mostrado para la confianza

    confianza = int(confianza)/100 # Conversion 0-1
    destino = session['destino']
    filename = session['filename']
    modelo = session['modelo']

    # Instancia del objeto Detector junto con el modelo y deteccion con el metodo inference
    try:
        dt = Detector(modelo, confianza)
        salida = dt.inference(destino)
        im = cv2.imread(destino)
        alturaImagen = im.shape[0]
        anchuraImagen = im.shape[1]

    # Extraccion de mascaras
        masks = dt.getOutputMask(salida)

        # Si no hay detecciones, generar el grafico igualmente pero sin alterar historico o registro
        if masks == 'Sin Defectos' or masks.size == 0: 
            fig = px.imshow(im)
            fig.update_traces(hoverinfo='skip')
            nombreGraph = "./static/uploads/graph_" + \
                filename[:-4] + "_" + fecha + ".html"
            fig.write_html(nombreGraph)

            session['resultado'] = nombreGraph
            session['widthHTML'] = anchuraImagen
            session['heightHTML'] = alturaImagen

            return render_template("deteccion.html",
                            mostrar_preview=False,
                            uploading=False,
                            mostrar_resultado=True,
                            conDefectos=False,
                            result=session['resultado'],
                            anchoGrafico=anchuraImagen,
                            altoGrafico=alturaImagen)

    except Exception as e:
        print('-------------------------------------------------------------------------')
        print('Excepcion al instanciar Detector, Esta presente el fichero model-x.x.pth?')
        print('Si no es asi ejecutar descargaModelo.py en el actual directorio')
        print('-------------------------------------------------------------------------')
        raise
        

    # Procesado de las mascaras binarias generadas y generacion del grafico Plotly 
    try:
        fig = px.imshow(im)
        fig.update_traces(hoverinfo='skip')
        listTipos = list()
        confianzas = dt.getConfidence(salida)
        maskTotal = np.zeros_like(masks[0])
        numeroDefecto = 0

        # Se recorren las mascaras de cada error y se obtiene el area de cada una con regionprops
        for i in range(len(masks)):
            numeroDefecto +=1
            mask = masks[i]
            maskTotal = maskTotal + mask
            labels = measure.label(mask)
            props = measure.regionprops(labels, mask)
            label = props[0].label
            contour = measure.find_contours(labels == label, 0.5)[0] # Contorno del defecto binario
            y, x = contour.T

            # Se obtiene el atributo area y se clasifica segun tamaño
            area = getattr(props[0], "area")
            if area >= 200:
                tipo = ('Big')
                listTipos.append(tipo)
            elif area >= 100:
                tipo = ('Medium')
                listTipos.append(tipo)
            else:
                tipo = ('Small')
                listTipos.append(tipo)

            # Informacion mostrada al pasar el cursor por el defecto actual
            hoverinfo = ''
            hoverinfo += f'<b>{"Área"}: {area:.2f}</b><br>'
            hoverinfo += f'<b>{"Confianza"}: {"{:.2%}".format(confianzas[i])}</b><br>' 
            hoverinfo += f'<b>{"Tipo"}: {tipo}</b><br>'
            fig.add_trace(go.Scatter(
                x=x, y=y, name=numeroDefecto,
                mode='lines', fill='toself', showlegend=True,
                hovertemplate=hoverinfo, hoveron='points+fills'))
    except Exception as e:
        print('-----------------------------------------------------')
        print('Excepcion al generar mascaras, Ha habido detecciones?')
        print('-----------------------------------------------------')
        raise
    
    nombreGraph = "./static/uploads/graph_" + \
        filename[:-4] + "_" + fecha + ".html"
    fig.write_html(nombreGraph)
    
    # Como control se comprueba que los valores esten entre 0-1
    maskTotal[maskTotal > 1] = 1
    mascaraImagen = Image.fromarray(maskTotal)

    session['mascaraBinaria'] = './static/uploads/'+filename+'MB.png'
    mascaraImagen.save('./static/uploads/'+filename+'MB.png')

    # Generando registro en CSV
    numErrores = Counter(listTipos)

    # Dependiendo de la existencia del CSV, se añade fila, acumula en una existente o se crea dicho fichero
    if os.path.isfile('registro.csv'):
        df = pd.read_csv('registro.csv', index_col=0).copy()
        if fechaRegistro in df.Fecha.unique():
            for i in range(len(df.Fecha)):
                if df.Fecha[i] == fechaRegistro:
                    df.iat[i, 0] += 1  
                    df.iat[i, 2] += numErrores['Small']
                    df.iat[i, 3] += numErrores['Medium']
                    df.iat[i, 4] += numErrores['Big']
            df.to_csv(r'registro.csv')

        else:
            df = pd.read_csv('registro.csv', index_col=0).copy()
            data = [{'N':1,'Fecha':fechaRegistro, 'Small':numErrores['Small'], 'Medium':numErrores['Medium'], 'Big':numErrores['Big']}]
            df = df.append(data, ignore_index=True,sort=False)
            df.to_csv(r'registro.csv')

    else:
        data = [{'N':1,'Fecha':fechaRegistro, 'Small':numErrores['Small'], 'Medium':numErrores['Medium'], 'Big':numErrores['Big']}]
        df = pd.DataFrame(data)
        df.to_csv(r'registro.csv')

    # Generacion del grafico de lineas Plotly
    dfc = pd.read_csv('registro.csv')
    df = dfc.copy()
    tope = df.iloc[:,2:5].sum(axis=1).max() + 5 # Valor mas alto en todos los ejes y para ajustar el tamaño del grafico, margen de 5

    fig = go.Figure()

    areaSmall = go.Scatter(
     x= df['Fecha'], y = df['Small'],
     name = 'Defectos Pequeños',
     mode = 'lines',
     line=dict(width=0.5, color='orange'),
     stackgroup = 'one')

    areaMedium = go.Scatter(
        x= df['Fecha'], y = df['Medium'],
        name = 'Defectos Medianos',
        mode = 'lines',
        line=dict(width=0.5, color='green'),
        stackgroup = 'one')

    areaBig= go.Scatter(
        x= df['Fecha'], y = df['Big'],
        name = 'Defectos Grandes',
        mode = 'lines',
        line=dict(width=0.5, color='red'),
        stackgroup = 'one')

    linea = go.Scatter(
        x=df['Fecha'],
        y=df['N'],
        mode='lines+markers',
        name='Imagenes Procesadas',
        line=dict(width=1, color='blue', dash="dashdot"),
        yaxis='y2'
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(areaSmall)
    fig.add_trace(areaMedium)
    fig.add_trace(areaBig)
    fig.add_trace(linea,secondary_y=True)

    fig.update_layout(title = "Histórico de Defectos", hovermode="x")

    fig.update_xaxes(
        title_text = 'Fecha',
        title_font=dict(size=16, family='Verdana', color='black'),
        tickfont=dict(family='Calibri', color='darkred', size=14))

    fig.update_yaxes(
        range=[0,tope],
        title_text = "Defectos",
        title_font=dict(size=16, family='Verdana', color='black'),
        tickfont=dict(family='Calibri', color='darkred', size=14))

    fig.update_yaxes(
        range=[0,tope],
        title_text = "Imagenes Procesadas",     
        title_font=dict(size=16, family='Verdana', color='blue'),
        tickfont=dict(family='Calibri', color='blue', size=14), secondary_y=True)

    nombreHistorico = "./static/uploads/graficoHistorico.html"
    fig.write_html(nombreHistorico)

    session['resultado'] = nombreGraph
    session['widthHTML'] = anchuraImagen
    session['heightHTML'] = alturaImagen

    return render_template("deteccion.html",
                           mostrar_preview=False,
                           uploading=False,
                           mostrar_resultado=True,
                           conDefectos = True,
                           result=session['resultado'],
                           anchoGrafico=anchuraImagen,
                           altoGrafico=alturaImagen)

# Se acede al contenido de la variable sesion que apunta al grafico generado y se envia para descarga
@app.route("/descargarDinamico")
def descargarDinamico():
    ficheroHTML = session['resultado']
    filename = session['filename']
    return send_file(ficheroHTML, attachment_filename=filename[:-4] + ".html", as_attachment=True)

# Se compone la imagen original junto con la mascara binaria y se descarga
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

    plt.subplots_adjust(wspace=0, hspace=0) # Se elmiinan margenes

    tempFig = BytesIO() # Con BytesIO se almacena en memoria en lugar de tener que generar un fichero cada vez, ahorrando espacio

    plt.savefig(tempFig, pad_inches=0, bbox_inches='tight', format='png')
    tempFig.seek(0)
    return send_file(tempFig, mimetype="image/png", attachment_filename=filename[:-4] + "_CMP.png", as_attachment=True)

# Descarga de la mascara binaria
@app.route("/descargarMascara")
def descargarMascara():
    mascaraBinaria = session['mascaraBinaria']
    filename = session['filename']
    return send_file(mascaraBinaria, attachment_filename=filename[:-4] + "_MB.png", as_attachment=True)

# Carga de la seccion del historico
@app.route("/hist")
def hist():
    return render_template("historico.html", urlGraph='./static/uploads/graficoHistorico.html')

# Carga del FAQ de la web
@app.route("/faq")
def faq():
    return render_template("faq.html")

# Se comprueba la version local del modelo presente en la aplicacion
@app.route('/checkModelVersion', methods=['POST'])
def checkModelVersion():
    ficheros = os.listdir()
    ficheros = [m for m in ficheros if 'modelo' in m and '.pth' in m] # Debe de tener la palabra y extension adecuados
    
    versionLocalMasReciente, versionLocalMasRecienteTxt = getMostRecentVersion(
        ficheros)

    if os.path.exists("modelos.json"):
        os.remove("modelos.json") # Si existe el fichero y puede no estar actualizado, se elimina 

    ficheroJson = wget.download('https://raw.githubusercontent.com/fyi0000/TFG-GII-20.04/main/modelos.json')
    with open("modelos.json", "r") as fich:
        dictModelos = json.load(fich)
    
    # Se elmina el punto entre las versiones del formato 0.3 quedando solo 3, que es inferior a 1.0 que quedaria 10
    versiones = [int(v['version'].replace('.','')) for v in dictModelos['modelos']] 
    versionesTxt = [v['version'] for v in dictModelos['modelos']]
    
    lastVersion = max(versiones)
    lastVersionTxt = versionesTxt[versiones.index(lastVersion)]
    url = str(dictModelos['modelos'][versiones.index(lastVersion)]['url'])

    # Si la version local es igual o posterior a la version remota, se notifica que esta actualizado
    if versionLocalMasReciente >= lastVersion:
        nombreFichero = 'modelo-'+lastVersionTxt+'.pth'
        session['modelo'] = nombreFichero
        return jsonify(status = 'Updated', recentVersion=lastVersionTxt, oldVersion = versionLocalMasRecienteTxt)
    
    # Por el contrario, se actualiza el nombre del modelo a descargar para la actualizacion
    session['urlUpdate'] = url
    session['versionUpdate'] = lastVersionTxt
    nombreFichero = 'modelo-'+versionLocalMasRecienteTxt+'.pth'
    session['modelo'] = nombreFichero
    return jsonify(status='Outdated', recentVersion=lastVersionTxt, oldVersion = versionLocalMasRecienteTxt)

# Se actualiza a la ultima version el modelo
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

# Metodo auxiliar que busca en una lista el valor numerico mas alto de version y devuelve el mismo y en cadena
def getMostRecentVersion(list):
    masReciente = 0
    masRecienteTxt = ''
    for e in list:
        version = re.findall("[\\b\d\\b]", e) # Expresion regular que detecta del nombre modelo-X.X la seccion X.X
        versionNumber = int(''.join(version))
        if versionNumber > masReciente:
            masReciente = versionNumber
            masRecienteTxt = '.'.join(version)

    return masReciente, masRecienteTxt

# Ejecucion en el puerto local indicado
if __name__ == "__main__":
    app.run(host='0.0.0.0')
