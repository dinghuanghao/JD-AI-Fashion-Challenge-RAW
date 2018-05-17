import tensorflow as tf

import config
from model import resnet50
from util import data_loader

IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 50

estimator = resnet50.get_estimator(config.IMAGE_SHAPE)


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


i = 0
while i < config.EPOCH:
    i += 1
    try:
        estimator.train(
            input_fn=data_loader.dataset_input_fn,
            hooks=[MetricHook()]
        )
    except tf.errors.OutOfRangeError as e:
        print("epoch %d training over" % i)
print("train over")


