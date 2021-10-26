from object_detection import model_lib_v2


# dataset needs to be in TFRecord format

class FasterRCNNTrainer:
    def __init__(self):
        self.model_config = "/unmasked/data/pipeline.config"
        self.model_dir = "/model_meta/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8/"
        self.ckpt_dir = "/model_meta/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8/checkpoint"

    def train(self):
        """
        Must change values in config file to be accurate before this method works
        :return:
        """
        model_lib_v2.eval_continuously(
            pipeline_config_path=self.model_config,
            model_dir=self.model_dir,
            train_steps=500,
            sample_1_of_n_eval_examples=None,
            sample_1_of_n_eval_on_train_examples=5,
            checkpoint_dir=self.ckpt_dir,
            wait_interval=300, timeout=3600)


