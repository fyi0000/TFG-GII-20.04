"""
Autor: Fco Javier Yagüe Izquierdo
Año: 2021
Versión: 1.0

Descripción:
Aplicación Web en Flask que junto con Detectron 2 y el modelo entrenado para el mismo, procesa y aplica una detección
sobre una imagen facilitada por el usuario.
"""

from flask import Flask

# Instanciando la clase Flask con el nombre de la app
app = Flask(__name__)

# El decorador route indica que se reescribe el comportamiento de una funcion que ya existe
@app.route('/')

def hola_mundo():
    return 'Hola Mundo!!'