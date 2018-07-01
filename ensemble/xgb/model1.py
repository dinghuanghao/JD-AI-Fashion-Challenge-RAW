import os

from util import ensemble_util

model_config = ensemble_util.EnsembleConfig(model_path=os.path.abspath(__file__),
                                            corr_threshold=0.9,
                                            search=20,
                                            top_n=5,
                                            all_label=True)
