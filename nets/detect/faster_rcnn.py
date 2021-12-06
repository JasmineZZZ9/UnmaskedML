import model_lib_v2_test
import os

# dataset needs to be in TFRecord format

class FasterRCNNTrainer:
    def __init__(self):
        self.model_config = "/unmasked/data/pipeline.config"
        self.ckpt_dir = "/model_meta/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8"

    def train(self):
        """
        Must change values in config file to be accurate before this method works
        :return:
        """
        model_lib_v2_test.train_loop(
            pipeline_config_path=self.model_config,
            model_dir=self.ckpt_dir,
            train_steps=25000,
            checkpoint_every_n=200,
            checkpoint_max_to_keep=250)

    def evaluate(self):
        model_lib_v2_test.eval_continuously(
            pipeline_config_path=self.model_config,
            model_dir=self.ckpt_dir,
            train_steps=500,
            sample_1_of_n_eval_examples=1,
            checkpoint_dir=os.path.join(self.ckpt_dir, "point"),
            wait_interval=300, timeout=3600)


if __name__ == "__main__":
    trainer = FasterRCNNTrainer()
    trainer.train()
