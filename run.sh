docker run --rm --mount type=bind,source=`pwd`,target=/unmasked -p 8888:8888 -p 0.0.0.0:6006:6006 --gpus=all -i -t unmasked-main-gpu

