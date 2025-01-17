FROM gcperkins/unmaskedml-base-gpu
WORKDIR /unmasked
RUN apt-get install protobuf-compiler python-pil python-lxml -y
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

RUN pip3 install pip==19.3
RUN pip3 install cython # dependency for COCO API

# download COCO API for evaluation
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git
RUN cd cocoapi/PythonAPI && \
    make && \
    cp -r pycocotools /tensorflow/models/research/

# compile object detection API for transfer learning
RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    pip3 install .

RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
