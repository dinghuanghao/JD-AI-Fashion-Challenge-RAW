import tensorflow as tf

import config
from model import resnet50
from util import data_loader

estimator = resnet50.get_estimator(config.MODEL_CONFIG_1)


class MetricHook(tf.train.SessionRunHook):
    def after_create_session(self, session: tf.Session, coord):
        predicts = session.graph.get_tensor_by_name("my_output/Sigmoid:0")
        labels = session.graph.get_tensor_by_name("IteratorGetNext:1")
        # accuracy = tf.metrics.accuracy(labels, predicts)
        # tf.summary.scalar(accuracy)

    def begin(self):
        predictions = tf.get_default_graph().get_tensor_by_name("my_output/Sigmoid:0")
        labels = tf.get_default_graph().get_tensor_by_name("IteratorGetNext:1")
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        tf.summary.scalar("accuracy", accuracy[1])
        pass

    def end(self, session):
        pass

    def after_run(self, run_context, run_values):
        pass

    def before_run(self, run_context):
        pass


model_config = config.MODEL_CONFIG_1

try:
    estimator.train(
        input_fn=lambda: data_loader.training_input_fn(model_config),
        hooks=[MetricHook()]
    )
except tf.errors.OutOfRangeError as e:
    pass
print("train over")

