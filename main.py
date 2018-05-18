import tensorflow as tf
import os

from model.resnet4 import resnet50
from util import data_loader

estimator = resnet50.get_estimator()
model_config = resnet50.MODEL_CONFIG


class MetricHook(tf.train.SessionRunHook):
    def begin(self):
        predictions = tf.get_default_graph().get_tensor_by_name(model_config.output_tensor_name)
        labels = tf.get_default_graph().get_tensor_by_name("IteratorGetNext:1")
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        tf.summary.scalar("accuracy", accuracy[1])
        pass

model_config.save_before_train()

try:
    estimator.train(
        input_fn=lambda: data_loader.training_input_fn(model_config),
        hooks=[MetricHook()]
    )
except tf.errors.OutOfRangeError as e:
    pass

model_config.save_after_train()

print("train over")
