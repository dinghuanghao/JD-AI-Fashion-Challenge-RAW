import tensorflow as tf

from model.densenet1 import densenet121
from util import data_loader
from util import metrics


def train_evaluate(estimator, model_config):
    class MetricHook(tf.train.SessionRunHook):
        def begin(self):
            predictions = tf.get_default_graph().get_tensor_by_name(model_config.output_tensor_name)
            labels = tf.get_default_graph().get_tensor_by_name("IteratorGetNext:1")
            f2_score = metrics.smooth_f2_score(labels, predictions)
            tf.summary.scalar("train/f2-score", f2_score)
            pass

    model_config.save_before_train()

    print("training steps per epoch is : %d" % data_loader.get_max_step(model_config))
    print("training epoch is :%d" % model_config.epoch)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: data_loader.data_input_fn(model_config),
        max_steps=data_loader.get_max_step(model_config),
        hooks=[MetricHook()])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: data_loader.data_input_fn(model_config, True))

    for i in range(model_config.epoch):
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print("epoch %d train and evaluate over !" % i)

    model_config.save_after_train()
