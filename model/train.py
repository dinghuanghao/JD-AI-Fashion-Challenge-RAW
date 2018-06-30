import os
import sys

sys.path.append(os.path.abspath("../"))

from model.densenet201 import model28_val1


model28_val1.train()