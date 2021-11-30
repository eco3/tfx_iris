from typing import List, Text
from absl import logging
from os import path

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
import keras_tuner as kt

from tfx import v1 as tfx
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
from tfx_bsl.public import tfxio

_FEATURE_KEYS = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
_LABELS = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
_LABEL_KEY = 'class'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

_DNN_HIDDEN_LAYER_0 = 'dnn_hidden_layer_0'
_DNN_HIDDEN_LAYER_1 = 'dnn_hidden_layer_1'

_DNN_HIDDEN_LAYERS = [_DNN_HIDDEN_LAYER_0, _DNN_HIDDEN_LAYER_1]


def _make_serving_signatures(model, tf_transform_output: tft.TFTransformOutput):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(_LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        confidence = model(transformed_features)

        indices = tf.where(confidence)
        last_index = indices.get_shape().as_list()[1] - 1
        last_indices_value = tf.slice(indices, [0, last_index], [-1, -1])
        classes_shape = tf.reshape(last_indices_value, tf.shape(confidence))
        class_names = tf.gather(_LABELS, classes_shape)

        return {
            'confidence': confidence,
            'labels': class_names
        }

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return {
        'serving_default': serve_tf_examples_fn,
        'transform_features': transform_features_fn
    }


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
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        tf_transform_output.raw_metadata.schema
    )

    transform_layer = tf_transform_output.transform_features_layer()

    def apply_transform(raw_features):
        return _apply_preprocessing(raw_features, transform_layer)

    return dataset.map(apply_transform).repeat()


def _build_keras_model(hp: kt.HyperParameters) -> tf.keras.Model:
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
    d = keras.layers.concatenate(inputs)
    for layer in _DNN_HIDDEN_LAYERS:
        d = keras.layers.Dense(int(hp.get(layer)), activation='relu')(d)
    outputs = keras.layers.Dense(len(_LABELS), activation='softmax')(d)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.summary(print_fn=logging.info)
    return model


def _get_hyperparams() -> kt.HyperParameters:
    hp = kt.HyperParameters()

    hp.Int(_DNN_HIDDEN_LAYER_0, min_value=100, max_value=150)
    hp.Int(_DNN_HIDDEN_LAYER_1, min_value=50, max_value=70)

    return hp


# TFX Transform will call this function.
def preprocessing_fn(inputs):
    outputs = {}

    for key in _FEATURE_KEYS:
        outputs[key] = tft.scale_to_z_score(inputs[key])

    table_keys = _LABELS
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys,
        values=tf.cast(tf.range(len(table_keys)), tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64
    )
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    outputs[_LABEL_KEY] = table.lookup(inputs[_LABEL_KEY])

    return outputs


# TFX Tuner will call this function.
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    working_dir = path.join(
        fn_args.working_dir[0:(len('Tuner') + fn_args.working_dir.index('Tuner'))],
        "tensorboards",
        path.split(fn_args.working_dir.strip("/"))[-1]
    )

    tuner = kt.RandomSearch(
        _build_keras_model,
        max_trials=6,
        hyperparameters=_get_hyperparams(),
        allow_new_entries=False,
        objective="val_accuracy",
        directory=working_dir,
        project_name='iris_tune_model',
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = _input_fn(fn_args.train_files,
                              fn_args.data_accessor,
                              tf_transform_output,
                              batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files,
                             fn_args.data_accessor,
                             tf_transform_output,
                             batch_size=_TRAIN_BATCH_SIZE)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'callbacks': [keras.callbacks.TensorBoard(working_dir)],
        })


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
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

    if fn_args.custom_config and ('tensorboard_dir' in fn_args.custom_config.keys()) and fn_args.custom_config[
        'tensorboard_dir']:
        tensorboard_dir = fn_args.custom_config['tensorboard_dir']
    else:
        tensorboard_dir = fn_args.model_run_dir
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, update_freq='batch')

    hyperparameters = fn_args.hyperparameters

    if hyperparameters:
        if type(hyperparameters) is dict and 'values' in hyperparameters.keys():
            hyperparameters = hyperparameters['values']
    else:
        hyperparameters = kt.HyperParameters()
        hyperparameters.Fixed(_DNN_HIDDEN_LAYER_0, 100)
        hyperparameters.Fixed(_DNN_HIDDEN_LAYER_1, 50)

    model = _build_keras_model(hyperparameters)
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    signatures = _make_serving_signatures(model, tf_transform_output)

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
