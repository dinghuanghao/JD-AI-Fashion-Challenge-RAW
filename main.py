import tensorflow as tf

from model.resnet4 import resnet50
from util import data_loader
from util import metrics

estimator = resnet50.get_estimator()
model_config = resnet50.MODEL_CONFIG


class MetricHook(tf.train.SessionRunHook):
    def begin(self):
        predictions = tf.get_default_graph().get_tensor_by_name(model_config.output_tensor_name)
        labels = tf.get_default_graph().get_tensor_by_name("IteratorGetNext:1")
        f2_score = metrics.f2_score(labels, predictions)
        tf.summary.scalar("train/f2-score", f2_score)
        pass


def evaluate():
    result = estimator.evaluate(
        input_fn=lambda: data_loader.data_input_fn(model_config, validation=True),
        hooks=[MetricHook()]
    )
    for i in result:
        print(i)
    print("evaluate over")

def train():
    model_config.save_before_train()

    try:
        estimator.train(
            input_fn=lambda: data_loader.data_input_fn(model_config),
            hooks=[MetricHook()]
        )
    except tf.errors.OutOfRangeError as e:
        pass

    model_config.save_after_train()
    print("train over")

def train_evaluate():
    model_config.save_before_train()
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: data_loader.data_input_fn(model_config),
        hooks=[MetricHook()])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda : data_loader.data_input_fn(model_config, True),
        hooks=[MetricHook()])
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    model_config.save_after_train()
    print("train and evaluate over")

if __name__ == "__main__":
    # evaluate()
    # train()
    train_evaluate()