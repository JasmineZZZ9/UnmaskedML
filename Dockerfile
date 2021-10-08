FROM gcperkins/unmaskedml-base:edge
WORKDIR /unmasked
RUN apt-get install protobuf-compiler python-pil python-lxml -y
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

RUN pip3 install cython

RUN cd / && \
    mkdir -p /model_meta && \
    cd /model_meta && \
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz && \
    tar xzf faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz

RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git
RUN cd cocoapi/PythonAPI && \
    make && \
    cp -r pycocotools /tensorflow/models/research/
RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    pip3 install --use-feature=2020-resolver .

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
