STRATEGY_F2_FIRST = "f2-first"
STRATEGY_RELEVANCE_FIRST = "relevance-first"


class EnsembleConfig(object):
    """
    1. 将选择的模型保存到本地，且如果本地有模型选择记录，则直接使用本地记录
    2. 支持多种模型选择技巧：
        （1）最优的前N个
        （2）前M个中，相关性最低的前N个
    3. 只选择与输出相同的标签（N * 1），还是所有标签（N * 13）
    """

    def __init__(self,
                 model_path: str,
                 strategy=STRATEGY_F2_FIRST,
                 top_n=5,
                 all_label=False,
                 debug=False):
        pass


class ModelEnsemble(object):
    """
    1. 自动做K-FOLD训练，将所有的模型都进行保存， 并将预测结果和评估结果也进行保存
    """

    def __init__(self, config: EnsembleConfig):
        pass

    def train_all_label(self):
        pass

    def train_single_label(self, label):
        pass
