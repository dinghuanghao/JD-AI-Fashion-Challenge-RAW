import os
import sys

sys.path.append(os.path.abspath("../"))

from model.densenet201 import model25_val1


model25_val1.train()