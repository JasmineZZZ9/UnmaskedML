FROM gcperkins/unmaskedml-base-gpu:latest
WORKDIR /unmasked
RUN apt-get install protobuf-compiler python-pil python-lxml -y
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

RUN pip3 install pip==19.3
RUN pip3 install cython # dependency for COCO API

# download model weights and checkpoint for Faster RCNN Inception
RUN cd / && \
    mkdir -p /model_meta && \
    cd /model_meta && \
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz && \
    tar xzf faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz

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

RUN mkdir -p /ckpt

RUN mv /model_meta/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/checkpoint/ /model_meta/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/point/

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt