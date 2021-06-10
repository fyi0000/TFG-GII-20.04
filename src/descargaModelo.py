# Sencillo script que descarga el modelo inicial desde Google Drive y se asegura que el nombre sea correcto

import gdown

url="https://drive.google.com/uc?id=1Pt8Q7kT1ytPk9gBrHBpQBOlfKS1NEUmK"

gdown.download(url, output='./modelo-0.1.pth', quiet=True)