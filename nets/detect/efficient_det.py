import tensorflow as tf
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)


class EfficientDet:
    def __init__(self, model_path="../../data/export1/saved_model", label_map_path="../../data/map.pbtxt"):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("Loading model...")
        start_time = time.time()
        self.model = tf.saved_model.load(model_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! Took {elapsed_time} seconds")

        self.label_map = label_map_util.create_category_index_from_labelmap(label_map_path=label_map_path,
                                                                            use_display_name=True)

    def predict_image(self, img_path, threshold=.3):
        print('Running inference for {}... '.format(img_path), end='')

        img = np.array(Image.open(img_path))
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor)

        nb_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :nb_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = nb_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']
        # apply score threshold on detections
        boxes = boxes[scores > threshold]
        classes = classes[scores > threshold]
        scores = scores[scores > threshold]

        return img, boxes, classes, scores

    def annotate_img(self, img, boxes, classes, scores):
        annotated_img = img.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            annotated_img,
            boxes,
            classes,
            scores,
            self.label_map,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0,
            agnostic_mode=False)
        return annotated_img

    def visualize_prediction(self, img_path):
        img, boxes, classes, scores = self.predict_image(img_path)
        annotated_img = self.annotate_img(img, boxes, classes, scores)
        plt.figure()
        plt.imshow(annotated_img)
        plt.savefig("annotated_" + img_path.split("/")[-1])

    def calculate_metrics(self, tfrecord_path="/unmasked/data/test.tfrecord"):
        tfrecord = tf.data.TFRecordDataset(tfrecord_path)

        for raw_record in tfrecord.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            filename = example.features.feature["image/filename"].bytes_list.value

            print(filename, type(filename))


if __name__ == "__main__":
    efficient_det = EfficientDet()
    efficient_det.calculate_metrics()
