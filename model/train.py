import os
import sys

sys.path.append(os.path.abspath("../"))

from model.inceptionv3 import model20_val2

model20_val2.train()