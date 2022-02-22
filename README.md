# TFG-GII-20.04 - Fco. Javier Yagüe Izquierdo - 2021 - Universidad de Burgos
Detección de defectos en piezas metálicas usando radiografías y Deep Learning

## Características
- Herramienta creada con [Detectron2]
- Red entrenada sobre imágenes etiquetadas por los tutores del proyecto
- Aplicación web basada en [Flask]
- Muestra de resultados de forma interactiva gracias a [Plotly]
- Registro de histórico de detecciones y posibilidad de descarga de resultados

![alt text](https://github.com/fyi0000/TFG-GII-20.04/blob/main/latex/img/preview.png?raw=true)

## Requisitos
Oficialmente Detectron2 no es compatible con Windows a pesar de que algunos usuarios han conseguido compatibilizar algunas depenencias
Por ello los requisitos son:
- Linux o MacOS con Python 3.6 o superior
- Pytorch junto con TochVision
- (Opcional) OpenCV pero que en este proyecto sí se ha utilizado para la visualización

Más información acerca de la instalación en la sección oficial de la [instalación de detectron2]
Para un despliegue más cómodo y compatible el proyecto se instala y ejecuta en una imagen [Docker]

[Detectron2]: <https://github.com/facebookresearch/detectron2>
[instalación de detectron2]: <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>
[Flask]: <https://flask.palletsprojects.com/en/2.0.x/>
[Plotly]: <https://plotly.com/>
[Docker]: <https://www.docker.com/>

## Instalación de la imagen

Con la aplicación de [Docker] correctamente instalada y ejecutándose, introducir el siguiente comando en el directorio donde se encuentre el fichero Dockerfile del repositorio.

```sh
docker build . -t nombreimagen
```

Una vez construida la imagen, puede demorarse según la conexión, generar un contenedor y asignarle el puerto 5000.
Posteriormente a través de la línea de comandos y con el contenedor activo comprobar la presencia del modelo mediante "ls", de no estar presente ejecutar:

```sh
sudo python descargaModelo.py
```

Y posteriormente ejecutar la aplicación:

```sh
sudo python app.py
```

Accesible normalmente en : 127.0.0.1:5000 o localhost:5000

