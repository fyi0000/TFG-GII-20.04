FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install --user torch==1.8 torchvision==0.9 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN git clone https://github.com/fyi0000/TFG-GII-20.04 detectron2_repo/proyecto

RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo


RUN cp -a ./proyecto/src/. .
RUN cp ./proyecto/requirements.txt .
RUN cp  ./proyecto/src/registro.csv .
RUN cp  ./proyecto/src/descargaModelo.py .

RUN pip install --user -r requirements.txt

ENV PORT 5000
EXPOSE 5000

RUN ["python", "descargaModelo.py"] 

#CMD ["python", "app.py"] 
