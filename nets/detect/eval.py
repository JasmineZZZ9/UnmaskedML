import copy
import os
import pprint
import time

import numpy as np
import tensorflow.compat.v1 as tf

from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils as vutils

MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP

def concat_replica_results(tensor_dict):
  new_tensor_dict = {}
  for key, values in tensor_dict.items():
    new_tensor_dict[key] = tf.concat(values, axis=0)
  return new_tensor_dict


def _compute_losses_and_predictions_dicts(
    model, features, labels,
    add_regularization_loss=True):
  """Computes the losses dict and predictions dict for a model on inputs.
  Args:
    model: a DetectionModel (based on Keras).
    features: Dictionary of feature tensors from the input dataset.
      Should be in the format output by `inputs.train_input` and
      `inputs.eval_input`.
        features[fields.InputDataFields.image] is a [batch_size, H, W, C]
          float32 tensor with preprocessed images.
        features[HASH_KEY] is a [batch_size] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] (optional) is a
          [batch_size, H, W, C] float32 tensor with original images.
    labels: A dictionary of groundtruth tensors post-unstacking. The original
      labels are of the form returned by `inputs.train_input` and
      `inputs.eval_input`. The shapes may have been modified by unstacking with
      `model_lib.unstack_batch`. However, the dictionary includes the following
      fields.
        labels[fields.InputDataFields.num_groundtruth_boxes] is a
          int32 tensor indicating the number of valid groundtruth boxes
          per image.
        labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor
          containing the corners of the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a float32
          one-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor
          containing groundtruth weights for the boxes.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          float32 tensor containing only binary values, which represent
          instance masks for objects.
        labels[fields.InputDataFields.groundtruth_instance_mask_weights] is a
          float32 tensor containing weights for the instance masks.
        labels[fields.InputDataFields.groundtruth_keypoints] is a
          float32 tensor containing keypoints for each box.
        labels[fields.InputDataFields.groundtruth_dp_num_points] is an int32
          tensor with the number of sampled DensePose points per object.
        labels[fields.InputDataFields.groundtruth_dp_part_ids] is an int32
          tensor with the DensePose part ids (0-indexed) per object.
        labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
          float32 tensor with the DensePose surface coordinates.
        labels[fields.InputDataFields.groundtruth_group_of] is a tf.bool tensor
          containing group_of annotations.
        labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
          k-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_track_ids] is a int32
          tensor of track IDs.
        labels[fields.InputDataFields.groundtruth_keypoint_depths] is a
          float32 tensor containing keypoint depths information.
        labels[fields.InputDataFields.groundtruth_keypoint_depth_weights] is a
          float32 tensor containing the weights of the keypoint depth feature.
    add_regularization_loss: Whether or not to include the model's
      regularization loss in the losses dictionary.
  Returns:
    A tuple containing the losses dictionary (with the total loss under
    the key 'Loss/total_loss'), and the predictions dictionary produced by
    `model.predict`.
  """
  model_lib.provide_groundtruth(model, labels)
  preprocessed_images = features[fields.InputDataFields.image]

  prediction_dict = model.predict(
      preprocessed_images,
      features[fields.InputDataFields.true_image_shape],
      **model.get_side_inputs(features))
  prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

  losses_dict = model.loss(
      prediction_dict, features[fields.InputDataFields.true_image_shape])
  losses = [loss_tensor for loss_tensor in losses_dict.values()]
  if add_regularization_loss:
    # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
    ## need to convert these regularization losses from bfloat16 to float32
    ## as well.
    regularization_losses = model.regularization_losses()
    if regularization_losses:
      regularization_losses = ops.bfloat16_to_float32_nested(
          regularization_losses)
      regularization_loss = tf.add_n(
          regularization_losses, name='regularization_loss')
      losses.append(regularization_loss)
      losses_dict['Loss/regularization_loss'] = regularization_loss

  total_loss = tf.add_n(losses, name='total_loss')
  losses_dict['Loss/total_loss'] = total_loss

  return losses_dict, prediction_dict

def prepare_eval_dict(detections, groundtruth, features):
  """Prepares eval dictionary containing detections and groundtruth.
  Takes in `detections` from the model, `groundtruth` and `features` returned
  from the eval tf.data.dataset and creates a dictionary of tensors suitable
  for detection eval modules.
  Args:
    detections: A dictionary of tensors returned by `model.postprocess`.
    groundtruth: `inputs.eval_input` returns an eval dataset of (features,
      labels) tuple. `groundtruth` must be set to `labels`.
      Please note that:
        * fields.InputDataFields.groundtruth_classes must be 0-indexed and
          in its 1-hot representation.
        * fields.InputDataFields.groundtruth_verified_neg_classes must be
          0-indexed and in its multi-hot repesentation.
        * fields.InputDataFields.groundtruth_not_exhaustive_classes must be
          0-indexed and in its multi-hot repesentation.
        * fields.InputDataFields.groundtruth_labeled_classes must be
          0-indexed and in its multi-hot repesentation.
    features: `inputs.eval_input` returns an eval dataset of (features, labels)
      tuple. This argument must be set to a dictionary containing the following
      keys and their corresponding values from `features` --
        * fields.InputDataFields.image
        * fields.InputDataFields.original_image
        * fields.InputDataFields.original_image_spatial_shape
        * fields.InputDataFields.true_image_shape
        * inputs.HASH_KEY
  Returns:
    eval_dict: A dictionary of tensors to pass to eval module.
    class_agnostic: Whether to evaluate detection in class agnostic mode.
  """

  groundtruth_boxes = groundtruth[fields.InputDataFields.groundtruth_boxes]
  groundtruth_boxes_shape = tf.shape(groundtruth_boxes)
  # For class-agnostic models, groundtruth one-hot encodings collapse to all
  # ones.
  class_agnostic = (
      fields.DetectionResultFields.detection_classes not in detections)
  if class_agnostic:
    groundtruth_classes_one_hot = tf.ones(
        [groundtruth_boxes_shape[0], groundtruth_boxes_shape[1], 1])
  else:
    groundtruth_classes_one_hot = groundtruth[
        fields.InputDataFields.groundtruth_classes]
  label_id_offset = 1  # Applying label id offset (b/63711816)
  groundtruth_classes = (
      tf.argmax(groundtruth_classes_one_hot, axis=2) + label_id_offset)
  groundtruth[fields.InputDataFields.groundtruth_classes] = groundtruth_classes

  label_id_offset_paddings = tf.constant([[0, 0], [1, 0]])
  if fields.InputDataFields.groundtruth_verified_neg_classes in groundtruth:
    groundtruth[
        fields.InputDataFields.groundtruth_verified_neg_classes] = tf.pad(
            groundtruth[
                fields.InputDataFields.groundtruth_verified_neg_classes],
            label_id_offset_paddings)
  if fields.InputDataFields.groundtruth_not_exhaustive_classes in groundtruth:
    groundtruth[
        fields.InputDataFields.groundtruth_not_exhaustive_classes] = tf.pad(
            groundtruth[
                fields.InputDataFields.groundtruth_not_exhaustive_classes],
            label_id_offset_paddings)
  if fields.InputDataFields.groundtruth_labeled_classes in groundtruth:
    groundtruth[fields.InputDataFields.groundtruth_labeled_classes] = tf.pad(
        groundtruth[fields.InputDataFields.groundtruth_labeled_classes],
        label_id_offset_paddings)

  use_original_images = fields.InputDataFields.original_image in features
  if use_original_images:
    eval_images = features[fields.InputDataFields.original_image]
    true_image_shapes = features[fields.InputDataFields.true_image_shape][:, :3]
    original_image_spatial_shapes = features[
        fields.InputDataFields.original_image_spatial_shape]
  else:
    eval_images = features[fields.InputDataFields.image]
    true_image_shapes = None
    original_image_spatial_shapes = None

  eval_dict = eval_util.result_dict_for_batched_example(
      eval_images,
      features[inputs.HASH_KEY],
      detections,
      groundtruth,
      class_agnostic=class_agnostic,
      scale_to_absolute=True,
      original_image_spatial_shapes=original_image_spatial_shapes,
      true_image_shapes=true_image_shapes)

  return eval_dict, class_agnostic


def eager_eval_loop(
    detection_model,
    configs,
    eval_dataset,
    use_tpu=False,
    postprocess_on_cpu=False,
    global_step=None,
    ):
  """Evaluate the model eagerly on the evaluation dataset.
  This method will compute the evaluation metrics specified in the configs on
  the entire evaluation dataset, then return the metrics. It will also log
  the metrics to TensorBoard.
  Args:
    detection_model: A DetectionModel (based on Keras) to evaluate.
    configs: Object detection configs that specify the evaluators that should
      be used, as well as whether regularization loss should be included and
      if bfloat16 should be used on TPUs.
    eval_dataset: Dataset containing evaluation data.
    use_tpu: Whether a TPU is being used to execute the model for evaluation.
    postprocess_on_cpu: Whether model postprocessing should happen on
      the CPU when using a TPU to execute the model.
    global_step: A variable containing the training step this model was trained
      to. Used for logging purposes.
  Returns:
    A dict of evaluation metrics representing the results of this evaluation.
  """
  print("EAGER EVAL LOOP")
  del postprocess_on_cpu
  train_config = configs['train_config']
  eval_input_config = configs['eval_input_config']
  eval_config = configs['eval_config']
  add_regularization_loss = train_config.add_regularization_loss

  is_training = False
  detection_model._is_training = is_training  # pylint: disable=protected-access
  tf.keras.backend.set_learning_phase(is_training)

  evaluator_options = eval_util.evaluator_options_from_eval_config(
      eval_config)
  batch_size = eval_config.batch_size

  class_agnostic_category_index = (
      label_map_util.create_class_agnostic_category_index())
  class_agnostic_evaluators = eval_util.get_evaluators(
      eval_config,
      list(class_agnostic_category_index.values()),
      evaluator_options)

  class_aware_evaluators = None
  if eval_input_config.label_map_path:
    class_aware_category_index = (
        label_map_util.create_category_index_from_labelmap(
            eval_input_config.label_map_path))
    class_aware_evaluators = eval_util.get_evaluators(
        eval_config,
        list(class_aware_category_index.values()),
        evaluator_options)

  evaluators = None
  loss_metrics = {}

  @tf.function
  def compute_eval_dict(features, labels):
    """Compute the evaluation result on an image."""
    # For evaling on train data, it is necessary to check whether groundtruth
    # must be unpadded.
    boxes_shape = (
        labels[fields.InputDataFields.groundtruth_boxes].get_shape().as_list())
    unpad_groundtruth_tensors = (boxes_shape[1] is not None
                                 and not use_tpu
                                 and batch_size == 1)
    groundtruth_dict = labels
    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    losses_dict, prediction_dict = _compute_losses_and_predictions_dicts(
        detection_model, features, labels, add_regularization_loss)
    prediction_dict = detection_model.postprocess(
        prediction_dict, features[fields.InputDataFields.true_image_shape])
    eval_features = {
        fields.InputDataFields.image:
            features[fields.InputDataFields.image],
        fields.InputDataFields.original_image:
            features[fields.InputDataFields.original_image],
        fields.InputDataFields.original_image_spatial_shape:
            features[fields.InputDataFields.original_image_spatial_shape],
        fields.InputDataFields.true_image_shape:
            features[fields.InputDataFields.true_image_shape],
        inputs.HASH_KEY: features[inputs.HASH_KEY],
    }
    return losses_dict, prediction_dict, groundtruth_dict, eval_features

  agnostic_categories = label_map_util.create_class_agnostic_category_index()
  per_class_categories = label_map_util.create_category_index_from_labelmap(
      eval_input_config.label_map_path)
  keypoint_edges = [
      (kp.start, kp.end) for kp in eval_config.keypoint_edge]

  strategy = tf.compat.v2.distribute.get_strategy()

  for i, (features, labels) in enumerate(eval_dataset):
    try:
      (losses_dict, prediction_dict, groundtruth_dict,
       eval_features) = strategy.run(
           compute_eval_dict, args=(features, labels))
    except Exception as exc:  # pylint:disable=broad-except
      tf.logging.info('Encountered %s exception.', exc)
      tf.logging.info('A replica probably exhausted all examples. Skipping '
                      'pending examples on other replicas.')
      break
    (local_prediction_dict, local_groundtruth_dict,
     local_eval_features) = tf.nest.map_structure(
         strategy.experimental_local_results,
         [prediction_dict, groundtruth_dict, eval_features])
    local_prediction_dict = concat_replica_results(local_prediction_dict)
    local_groundtruth_dict = concat_replica_results(local_groundtruth_dict)
    local_eval_features = concat_replica_results(local_eval_features)

    eval_dict, class_agnostic = prepare_eval_dict(local_prediction_dict,
                                                  local_groundtruth_dict,
                                                  local_eval_features)
    for loss_key, loss_tensor in iter(losses_dict.items()):
      losses_dict[loss_key] = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                              loss_tensor, None)
    if class_agnostic:
      category_index = agnostic_categories
    else:
      category_index = per_class_categories

    if i % 100 == 0:
      tf.logging.info('Finished eval step %d', i)

    use_original_images = fields.InputDataFields.original_image in features
    if (use_original_images and i < eval_config.num_visualizations):
      sbys_image_list = vutils.draw_side_by_side_evaluation_image(
          eval_dict,
          category_index=category_index,
          max_boxes_to_draw=100,
          min_score_thresh=.3,
          use_normalized_coordinates=False,
          keypoint_edges=keypoint_edges or None)
      for j, sbys_image in enumerate(sbys_image_list):
        tf.compat.v2.summary.image(
            name='eval_side_by_side_{}_{}'.format(i, j),
            step=global_step,
            data=sbys_image,
            max_outputs=eval_config.num_visualizations)
      if eval_util.has_densepose(eval_dict):
        dp_image_list = vutils.draw_densepose_visualizations(
            eval_dict)
        for j, dp_image in enumerate(dp_image_list):
          tf.compat.v2.summary.image(
              name='densepose_detections_{}_{}'.format(i, j),
              step=global_step,
              data=dp_image,
              max_outputs=eval_config.num_visualizations)

    if evaluators is None:
      if class_agnostic:
        evaluators = class_agnostic_evaluators
      else:
        evaluators = class_aware_evaluators

    for evaluator in evaluators:
      evaluator.add_eval_dict(eval_dict)

    for loss_key, loss_tensor in iter(losses_dict.items()):
      if loss_key not in loss_metrics:
        loss_metrics[loss_key] = []
      loss_metrics[loss_key].append(loss_tensor)

  eval_metrics = {}
  print("WRITING METRICS")

  for evaluator in evaluators:
    eval_metrics.update(evaluator.evaluate())
  for loss_key in loss_metrics:
    eval_metrics[loss_key] = tf.reduce_mean(loss_metrics[loss_key])

  eval_metrics = {str(k): v for k, v in eval_metrics.items()}
  print('Eval metrics at step %d' % global_step.numpy())
  for k in eval_metrics:
    tf.compat.v2.summary.scalar(k, eval_metrics[k], step=global_step)
    tf.logging.info('\t+ %s: %f', k, eval_metrics[k])
  return eval_metrics

def _ensure_model_is_built(model, input_dataset, unpad_groundtruth_tensors):
  """Ensures that model variables are all built, by running on a dummy input.
  Args:
    model: A DetectionModel to be built.
    input_dataset: The tf.data Dataset the model is being trained on. Needed to
      get the shapes for the dummy loss computation.
    unpad_groundtruth_tensors: A parameter passed to unstack_batch.
  """
  features, labels = iter(input_dataset).next()

  @tf.function
  def _dummy_computation_fn(features, labels):
    model._is_training = False  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(False)

    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    return _compute_losses_and_predictions_dicts(model, features, labels)

  strategy = tf.compat.v2.distribute.get_strategy()
  if hasattr(tf.distribute.Strategy, 'run'):
    strategy.run(
        _dummy_computation_fn, args=(
            features,
            labels,
        ))
  else:
    strategy.experimental_run_v2(
        _dummy_computation_fn, args=(
            features,
            labels,
        ))

def eval_continuously(
    pipeline_config_path,
    config_override=None,
    train_steps=2000,
    sample_1_of_n_eval_examples=1,
    sample_1_of_n_eval_on_train_examples=5,
    use_tpu=False,
    override_eval_num_epochs=True,
    postprocess_on_cpu=False,
    model_dir=None,
    checkpoint_dir=None,
    wait_interval=180,
    timeout=3600,
    eval_index=0,
    save_final_config=False,
    **kwargs):
  """Run continuous evaluation of a detection model eagerly.
  This method builds the model, and continously restores it from the most
  recent training checkpoint in the checkpoint directory & evaluates it
  on the evaluation data.
  Args:
    pipeline_config_path: A path to a pipeline config file.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override the config from `pipeline_config_path`.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    sample_1_of_n_eval_examples: Integer representing how often an eval example
      should be sampled. If 1, will sample all examples.
    sample_1_of_n_eval_on_train_examples: Similar to
      `sample_1_of_n_eval_examples`, except controls the sampling of training
      data for evaluation.
    use_tpu: Boolean, whether training and evaluation should run on TPU.
    override_eval_num_epochs: Whether to overwrite the number of epochs to 1 for
      eval_input.
    postprocess_on_cpu: When use_tpu and postprocess_on_cpu are true,
      postprocess is scheduled on the host cpu.
    model_dir: Directory to output resulting evaluation summaries to.
    checkpoint_dir: Directory that contains the training checkpoints.
    wait_interval: The mimmum number of seconds to wait before checking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait for a checkpoint. Execution
      will terminate if no new checkpoints are found after these many seconds.
    eval_index: int, If given, only evaluate the dataset at the given
      index. By default, evaluates dataset at 0'th index.
    save_final_config: Whether to save the pipeline config file to the model
      directory.
    **kwargs: Additional keyword arguments for configuration override.
  """
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
      'create_pipeline_proto_from_configs']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']

  configs = get_configs_from_pipeline_file(
      pipeline_config_path, config_override=config_override)
  kwargs.update({
      'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples,
      'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
  })
  if train_steps is not None:
    kwargs['train_steps'] = train_steps
  if override_eval_num_epochs:
    kwargs.update({'eval_num_epochs': 1})
    tf.logging.warning(
        'Forced number of epochs for all eval validations to be 1.')
  configs = merge_external_params_with_configs(
      configs, None, kwargs_dict=kwargs)
  if model_dir and save_final_config:
    tf.logging.info('Saving pipeline config file to directory {}'.format(
        model_dir))
    pipeline_config_final = create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config_final, model_dir)

  model_config = configs['model']
  train_input_config = configs['train_input_config']
  eval_config = configs['eval_config']
  eval_input_configs = configs['eval_input_configs']
  eval_on_train_input_config = copy.deepcopy(train_input_config)
  eval_on_train_input_config.sample_1_of_n_examples = (
      sample_1_of_n_eval_on_train_examples)
  if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
    tf.logging.warning('Expected number of evaluation epochs is 1, but '
                       'instead encountered `eval_on_train_input_config'
                       '.num_epochs` = '
                       '{}. Overwriting `num_epochs` to 1.'.format(
                           eval_on_train_input_config.num_epochs))
    eval_on_train_input_config.num_epochs = 1

  if kwargs['use_bfloat16']:
    tf.compat.v2.keras.mixed_precision.set_global_policy('mixed_bfloat16')

  eval_input_config = eval_input_configs[eval_index]
  strategy = tf.compat.v2.distribute.get_strategy()
  with strategy.scope():
    detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
        model_config=model_config, is_training=True)

  eval_input = strategy.experimental_distribute_dataset(
      inputs.eval_input(
          eval_config=eval_config,
          eval_input_config=eval_input_config,
          model_config=model_config,
          model=detection_model))

  global_step = tf.compat.v2.Variable(
      0, trainable=False, dtype=tf.compat.v2.dtypes.int64)

  optimizer, _ = optimizer_builder.build(
      configs['train_config'].optimizer, global_step=global_step)

  for i in range(6):
    ckpt_name = f"/unmasked/data/docker/ckpt-{i+1}"
    ckpt = tf.compat.v2.train.Checkpoint(
        step=global_step, model=detection_model, optimizer=optimizer)

    # We run the detection_model on dummy inputs in order to ensure that the
    # model and all its variables have been properly constructed. Specifically,
    # this is currently necessary prior to (potentially) creating shadow copies
    # of the model variables for the EMA optimizer.
    if eval_config.use_moving_averages:
      unpad_groundtruth_tensors = (eval_config.batch_size == 1 and not use_tpu)
      _ensure_model_is_built(detection_model, eval_input,
                             unpad_groundtruth_tensors)
      optimizer.shadow_copy(detection_model)

    ckpt.restore(ckpt_name).expect_partial()

    if eval_config.use_moving_averages:
      optimizer.swap_weights()

    summary_writer = tf.compat.v2.summary.create_file_writer(
        os.path.join(model_dir, 'eval'))
    print("FILE NAME GRANT", os.path.join(model_dir, 'eval'))
    with summary_writer.as_default():
      eager_eval_loop(
          detection_model,
          configs,
          eval_input,
          use_tpu=use_tpu,
          postprocess_on_cpu=postprocess_on_cpu,
          global_step=global_step,
          )
    summary_writer.flush()
if __name__ == "__main__":
    eval_continuously(
        pipeline_config_path="/unmasked/data/pipeline.config",
        model_dir="/unmasked/data/docker",
        train_steps=4000,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=(5),
        checkpoint_dir="unmasked/data/docker",
        wait_interval=300, timeout=3600)
