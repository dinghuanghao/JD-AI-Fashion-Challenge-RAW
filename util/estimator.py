import tensorflow as tf

from util import data_loader
from util import metrics


def train_evaluate(estimator, model_config):
    class SummaryMetricHook(tf.train.SessionRunHook):
        def begin(self):
            # labes name是通过data_loader.data_input_fn()函数打印获得
            # 在training过程中，labels和predictions处于同一张Graph
            labels = tf.get_default_graph().get_tensor_by_name("IteratorGetNext:1")
            predictions = tf.get_default_graph().get_tensor_by_name(model_config.output_tensor_name)
            f2_score = metrics.smooth_f2_score(labels, predictions)
            tf.summary.scalar("train/f2-score", f2_score)
            pass

    model_config.save_before_train()

    print("training steps per epoch is : %d" % data_loader.get_max_step(model_config))
    print("training epoch is :%d" % model_config.epoch)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: data_loader.data_input_fn(model_config),
        hooks=[SummaryMetricHook()])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: data_loader.data_input_fn(model_config, True))

    # 当train_spec用完一次之后，会进行evaluate，然后再次初始化train_spec
    # 因此除非设置了MaxSteps或者自己的钩子函数，否则永远不会停
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    model_config.save_after_train()
