from typing import List, Text
from absl import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

_FEATURE_KEYS = [
    'sepallength', 'sepalwidth', 'petallength', 'petalwidth'
]
_LABEL_KEY = 'class'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature.
    """
    outputs = {}

    # Uses features defined in _FEATURE_KEYS only.
    for key in _FEATURE_KEYS:
        # tft.scale_to_z_score computes the mean and variance of the given feature
        # and scales the output based on the result.
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # For the label column we provide the mapping from string to index.
    # We could instead use `tft.compute_and_apply_vocabulary()` in order to
    # compute the vocabulary dynamically and perform a lookup.
    # Since in this example there are only 3 possible values, we use a hard-coded
    # table for simplicity.
    table_keys = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  # TODO
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys,
        values=tf.cast(tf.range(len(table_keys)), tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64)
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    outputs[_LABEL_KEY] = table.lookup(inputs[_LABEL_KEY])

    return outputs


def _apply_preprocessing(raw_features, tft_layer):
    transformed_features = tft_layer(raw_features)
    if _LABEL_KEY in raw_features:
        transformed_label = transformed_features.pop(_LABEL_KEY)
        return transformed_features, transformed_label
    else:
        return transformed_features, None


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        tf_transform_output.raw_metadata.schema
    )

    transform_layer = tf_transform_output.transform_features_layer()

    def apply_transform(raw_features):
        return _apply_preprocessing(raw_features, transform_layer)

    return dataset.map(apply_transform).repeat()


def _build_keras_model() -> tf.keras.Model:
    inputs = [
        keras.layers.Input(shape=(1,), name=f)
        for f in _FEATURE_KEYS
    ]
    d = keras.layers.concatenate(inputs)
    for _ in range(2):
        d = keras.layers.Dense(10, activation='relu')(d)
    outputs = keras.layers.Dense(3)(d)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')

    model = _build_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    model.save(fn_args.serving_model_dir, save_format='tf')
